"""Gemini REST client; keys remain in memory/environment and never enter URLs/logs."""
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path


class ParentError(RuntimeError):
    def __init__(self, message, retryable=False):
        super().__init__(message)
        self.retryable = retryable


class GeminiParent:
    BASE = 'https://generativelanguage.googleapis.com/v1beta'

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
        encoded = json.dumps(body).encode() if body is not None else None
        headers = {'x-goog-api-key': self._key, 'Content-Type': 'application/json'}
        request = urllib.request.Request(self.BASE + suffix, data=encoded, headers=headers,
                                         method='POST' if body is not None else 'GET')
        for attempt in range(self.settings['max_retries'] + 1):
            started = time.monotonic()
            try:
                with self._open(request, timeout=self.settings['timeout_seconds']) as response:
                    result = json.load(response)
                self.logger({'operation': operation, 'model': self.model, 'attempt': attempt + 1,
                             'status': 'ok', 'duration_seconds': round(time.monotonic() - started, 3),
                             'usage': result.get('usageMetadata', {})})
                return result
            except urllib.error.HTTPError as exc:
                transient = exc.code == 429 or 500 <= exc.code < 600
                retry_after = exc.headers.get('Retry-After', '') if exc.headers else ''
                exc.close()
                # Provider bodies can reflect request data; log status codes only.
                self.logger({'operation': operation, 'model': self.model, 'attempt': attempt + 1,
                             'status': 'http_error', 'http_status': exc.code})
                if not transient or attempt == self.settings['max_retries']:
                    raise ParentError(f'Gemini HTTP {exc.code} during {operation}', transient) from None
                delay = min(30, float(retry_after)) if retry_after.isdigit() else min(30, 2 ** attempt)
                self._sleep(delay)
            except (urllib.error.URLError, TimeoutError, OSError):
                self.logger({'operation': operation, 'model': self.model, 'attempt': attempt + 1,
                             'status': 'network_error'})
                if attempt == self.settings['max_retries']:
                    raise ParentError(f'Gemini network failure during {operation}', True) from None
                self._sleep(min(30, 2 ** attempt))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ParentError('Gemini returned malformed transport JSON', True) from None
        raise ParentError('Gemini retries exhausted')

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
