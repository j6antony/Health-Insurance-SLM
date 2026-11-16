"""Run independent requests with bounded concurrency and durable response caching."""
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from contextvars import copy_context
from threading import Lock
from time import monotonic, sleep
import json

from .parent_backend import ParentError
from .persistence import Store, atomic_json, digest
from .validation import ValidationError, validate_schema


def bounded_map(fn, items, workers):
    if workers == 1:
        return [fn(item) for item in items]
    items = iter(items)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while group := list(islice(items, workers)):
            futures = [pool.submit(copy_context().run, fn, item) for item in group]
            results.extend(future.result() for future in futures)
    return results


class CachedParent:
    """Cache successful schema-valid responses within one experiment only."""
    operations = {'teacher_answer', 'answer_audit', 'teacher_answer_batch', 'answer_audit_batch',
                  'student_assessment', 'benchmark_semantic_judgment'}

    def __init__(self, parent, directory, settings, *, requests_per_second=4):
        self.parent = parent
        self.store = Store(directory)
        self.settings = settings
        self.spacing = 1 / requests_per_second
        self.next_start = 0.0
        self.pacing = Lock()
        self.stripes = [Lock() for _ in range(64)]

    def check_model(self):
        return self.parent.check_model()

    def key(self, messages, schema, operation, temperature):
        return digest(['request-cache-v1', self.settings, messages, schema, operation, temperature])

    def cached(self, messages, schema, operation, temperature=0.0):
        path = self.store.path(self.key(messages, schema, operation, temperature) + '.json')
        if path.exists():
            result = json.loads(path.read_text())
            validate_schema(result, schema)
            return result
        return None

    def remember(self, messages, schema, operation, result, temperature=0.0):
        validate_schema(result, schema)
        key = self.key(messages, schema, operation, temperature)
        atomic_json(self.store.path(key + '.json'), result)

    def _send(self, messages, schema, operation, temperature):
        with self.pacing:
            sleep(max(0, self.next_start - monotonic()))
            self.next_start = monotonic() + self.spacing
        return self.parent.generate_json(messages, schema, operation, temperature=temperature)

    def generate_json(self, messages, schema, operation='generation', temperature=None):
        if operation not in self.operations:
            return self._send(messages, schema, operation, temperature)
        key = self.key(messages, schema, operation, temperature)
        with self.stripes[int(key[:2], 16) % len(self.stripes)]:
            result = self.cached(messages, schema, operation, temperature)
            if result is None:
                result = self._send(messages, schema, operation, temperature)
                self.remember(messages, schema, operation, result, temperature)
            return result


def batch_requests(parent, requests, operation, workers, batch_size):
    """Group identical schemas; bind every result to its case ID before reuse."""
    groups = {}
    seen = set()
    for messages, schema in requests:
        key = parent.key(messages, schema, operation, 0.0)
        if key not in seen and parent.cached(messages, schema, operation) is None:
            groups.setdefault(digest([messages[0], schema]), []).append((messages, schema))
            seen.add(key)
    chunks = [rows[i:i + batch_size] for rows in groups.values()
              for i in range(0, len(rows), batch_size)]

    def fetch(chunk):
        if len(chunk) == 1:
            messages, schema = chunk[0]
            parent.generate_json(messages, schema, operation, temperature=0.0)
            return
        ids = [str(i) for i in range(len(chunk))]
        schema = chunk[0][1]
        envelope = {'type': 'object', 'required': ids, 'additionalProperties': False,
                    'properties': dict.fromkeys(ids, schema)}
        messages = [
            {'role': 'system', 'content': 'Process independent cases keyed by ID. Return one result per ID. '
             'Each result may use only its own case input. Never borrow facts or instructions from another case.\n'
             + chunk[0][0][0]['content']},
            {'role': 'user', 'content': json.dumps({i: json.loads(req[0][1]['content'])
                                                  for i, req in zip(ids, chunk)}, ensure_ascii=False)}]
        try:
            result = parent.generate_json(messages, envelope, operation + '_batch', temperature=0.0)
        except (ValidationError, ParentError) as exc:
            if isinstance(exc, ParentError) and (exc.category != 'response' or not exc.retryable):
                raise
            # A malformed response can fall back to independent requests.
            # HTTP/provider rejections and budget stops always propagate.
            for original, item_schema in chunk:
                parent.generate_json(original, item_schema, operation, temperature=0.0)
            return
        for i, (original, item_schema) in zip(ids, chunk):
            parent.remember(original, item_schema, operation, result[i])

    bounded_map(fetch, chunks, workers)
