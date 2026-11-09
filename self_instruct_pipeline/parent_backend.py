"""OpenAI/Gemini clients. Credentials stay in memory/environment, never URLs or logs."""
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from .persistence import now
from .budget import ACTIVE_BUDGET


class ParentError(RuntimeError):
    def __init__(self, message, retryable=False):
        super().__init__(message)
        self.retryable = retryable


class JSONParent:
    NAME = "Parent"

    def __init__(self, settings, *, api_key=None, logger=None, opener=None, sleep=time.sleep):
        self.settings = settings
        self.model = settings['model']
        self._key = api_key or os.environ.get(settings['api_key_env'])
        if not self._key:
            raise ParentError(f"Set {settings['api_key_env']} before calling the API")
        self.logger = logger or (lambda event: None)
        self._sleep = sleep
        self._open = opener or self._urlopen
        self.context = ssl.create_default_context()
        # Some macOS Python installations lack their OpenSSL CA bundle. Use the
        # system bundle with full certificate/hostname verification, never CERT_NONE.
        if not ssl.get_default_verify_paths().cafile and Path('/etc/ssl/cert.pem').is_file():
            self.context = ssl.create_default_context(cafile='/etc/ssl/cert.pem')

    def _urlopen(self, request, timeout):
        return urllib.request.urlopen(request, timeout=timeout, context=self.context)

    def _request(self, suffix, body=None, operation='request'):
        budget = ACTIVE_BUDGET.get()
        encoded = json.dumps(body).encode() if body is not None else None
        headers = {**self.auth_headers(), 'Content-Type': 'application/json'}
        request = urllib.request.Request(self.BASE + suffix, data=encoded, headers=headers,
                                         method='POST' if body is not None else 'GET')
        for attempt in range(self.settings['max_retries'] + 1):
            reservation = budget.reserve(self.settings, body) if budget else None
            started = time.monotonic()
            timestamp = now()
            def log(status, **fields):
                event = {'timestamp': timestamp, 'finished_at': now(),
                             'provider': self.settings['provider'], 'operation': operation,
                             'model': self.model, 'attempt': attempt + 1, 'status': status,
                             'duration_seconds': round(time.monotonic() - started, 3), **fields}
                self.logger(event)
                if budget:
                    budget.settle(reservation, event)
            try:
                with self._open(request, timeout=self.settings['timeout_seconds']) as response:
                    result = json.load(response)
            except urllib.error.HTTPError as exc:
                transient = exc.code == 429 or 500 <= exc.code < 600
                retry_after = exc.headers.get('Retry-After', '') if exc.headers else ''
                # Keep structured diagnostics, never the raw error body/message:
                # provider messages can echo source documents or credentials.
                details = {}
                try:
                    error = json.loads(exc.read(65536)).get('error', {})
                    if isinstance(error, dict):
                        for key in ('type', 'code', 'param'):
                            value = error.get(key)
                            if (isinstance(value, str) and self._key not in value
                                    and re.fullmatch(r'[A-Za-z0-9_.\[\]-]{1,100}', value)):
                                details['error_' + key] = value
                except (ValueError, TypeError, AttributeError, OSError):
                    pass
                finally:
                    exc.close()
                log('http_error', http_status=exc.code, **details)
                if not transient or attempt == self.settings['max_retries']:
                    diagnostic = ', '.join(f'{key}={value}' for key, value in details.items())
                    suffix = f' ({diagnostic})' if diagnostic else ''
                    raise ParentError(f'{self.NAME} HTTP {exc.code} during {operation}{suffix}', transient) from None
                delay = min(30, float(retry_after)) if retry_after.isdigit() else min(30, 2 ** attempt)
                self._sleep(delay)
            except (urllib.error.URLError, TimeoutError, OSError):
                log('network_error')
                if attempt == self.settings['max_retries']:
                    raise ParentError(f'{self.NAME} network failure during {operation}', True) from None
                self._sleep(min(30, 2 ** attempt))
            except (json.JSONDecodeError, UnicodeDecodeError):
                log('malformed_transport')
                raise ParentError(f'{self.NAME} returned malformed transport JSON', True) from None
            else:
                # A disk/logging error must never retry an already billed API call.
                log('ok', usage=result.get('usageMetadata', result.get('usage', {})),
                    response_id=result.get('id', ''), response_status=result.get('status', ''))
                return result
        raise ParentError(f'{self.NAME} retries exhausted')

class GeminiParent(JSONParent):
    NAME = 'Gemini'
    BASE = 'https://generativelanguage.googleapis.com/v1beta'

    def auth_headers(self):
        return {'x-goog-api-key': self._key}

    def check_model(self):
        result = self._request('/models/' + self.model, operation='model_check')
        if 'generateContent' not in result.get('supportedGenerationMethods', []):
            raise ParentError('Selected model does not support generateContent')
        return {'model': result['name'], 'output_token_limit': result.get('outputTokenLimit')}

    def generate_json(self, messages, response_schema, operation='generation', temperature=None):
        if len(messages) != 2 or [m['role'] for m in messages] != ['system', 'user']:
            raise ValueError('Backend expects exactly system and user messages')
        body = {
            'systemInstruction': {'parts': [{'text': messages[0]['content']}]},
            'contents': [{'role': 'user', 'parts': [{'text': messages[1]['content']}]}],
            'generationConfig': {
                'temperature': self.settings['temperature'] if temperature is None else temperature,
                'maxOutputTokens': self.settings['max_output_tokens'],
                'responseMimeType': 'application/json', 'responseJsonSchema': response_schema,
            },
        }
        result = self._request('/models/' + self.model + ':generateContent', body, operation)
        candidates = result.get('candidates', [])
        if len(candidates) != 1 or candidates[0].get('finishReason') != 'STOP':
            raise ParentError('Gemini response blocked, incomplete, or truncated', True)
        parts = candidates[0].get('content', {}).get('parts', [])
        text = ''.join(p.get('text', '') for p in parts if not p.get('thought'))
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            raise ParentError('Gemini returned invalid generated JSON', True) from None


class OpenAIParent(JSONParent):
    NAME = 'OpenAI'
    BASE = 'https://api.openai.com/v1'

    def auth_headers(self):
        return {'Authorization': f'Bearer {self._key}'}

    def check_model(self):
        result = self._request('/models/' + self.model, operation='model_check')
        if result.get('id') != self.model:
            raise ParentError('OpenAI did not confirm the configured model')
        return {'model': result['id'], 'provider': 'openai'}

    def generate_json(self, messages, response_schema, operation='generation', temperature=None):
        if len(messages) != 2 or [m['role'] for m in messages] != ['system', 'user']:
            raise ValueError('Backend expects exactly system and user messages')
        # JSON mode accepts an object at the root, while our pipeline also uses
        # arrays and open-ended scenario.context dictionaries. Wrap the result
        # in transport only, then enforce the ORIGINAL schema locally. This avoids
        # silently dropping context keys to fit OpenAI's strict-schema subset.
        instructions = messages[0]['content'] + (
            '\nTransport requirement for this API call: return one JSON object with '
            'exactly one key, "result". Its value is the JSON value requested above, '
            'which must match this JSON Schema. The wrapper is transport only; '
            'do not change the fields inside result.\n'
            + json.dumps(response_schema)
        )
        effort = self.settings.get('reasoning_effort', 'medium')
        body = {
            'model': self.model,
            'instructions': instructions,
            'input': [{'role': 'user', 'content': messages[1]['content']}],
            'text': {'format': {'type': 'json_object'}},
            'reasoning': {'effort': effort},
            'max_output_tokens': self.settings['max_output_tokens'],
            'store': False,
            'service_tier': 'default',
        }
        # GPT-6 accepts temperature only with reasoning disabled.
        if effort == 'none':
            body['temperature'] = self.settings['temperature'] if temperature is None else temperature
        result = self._request('/responses', body, operation)
        if result.get('status') != 'completed' or result.get('error') or result.get('incomplete_details'):
            raise ParentError('OpenAI response failed, incomplete, or truncated', True)
        parts = [part for item in result.get('output', []) if item.get('type') == 'message'
                 for part in item.get('content', [])]
        if any(part.get('type') == 'refusal' for part in parts):
            raise ParentError('OpenAI refused this request')
        text = ''.join(part.get('text', '') for part in parts if part.get('type') == 'output_text')
        try:
            wrapped = json.loads(text)
            if not isinstance(wrapped, dict) or set(wrapped) != {'result'}:
                raise ValueError('Expected transport wrapper')
            from .validation import validate_schema
            validate_schema(wrapped['result'], response_schema)
            return wrapped['result']
        except (ValueError, TypeError, KeyError):
            raise ParentError('OpenAI returned invalid or schema-incompatible JSON', True) from None


def create_parent(settings, **kwargs):
    if settings['provider'] == 'openai':
        return OpenAIParent(settings, **kwargs)
    if settings['provider'] == 'gemini':
        return GeminiParent(settings, **kwargs)
    raise ValueError('Unsupported parent provider')
