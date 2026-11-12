"""Durable per-attempt usage journal and Excel-readable CSV, without request content."""
import csv
import os
import json
import uuid
from decimal import Decimal
from pathlib import Path

from .persistence import Store, now, atomic_json, atomic_jsonl

# Standard global text rates, USD / million tokens; verified 2026-11-09.
# Update this table when changing models or when provider prices change.
PRICING_SOURCE = 'https://developers.openai.com/api/docs/models/gpt-6-luna'
RATES = {('openai', 'gpt-6-luna'): ('0.10', '0.01', '0.50', '0.125')}
FIELDS = ['event_id', 'run_id', 'timestamp', 'finished_at', 'provider', 'model', 'operation', 'attempt',
          'status', 'http_status', 'error_type', 'error_code', 'error_param', 'error_hint', 'response_status', 'response_id', 'duration_seconds',
          'input_tokens', 'cached_input_tokens', 'cache_write_tokens', 'output_tokens',
          'reasoning_tokens', 'total_tokens', 'estimated_cost_usd', 'cost_status',
          'input_usd_per_million', 'cached_input_usd_per_million',
          'output_usd_per_million', 'cache_write_usd_per_million', 'pricing_verified',
          'pricing_source', 'cost_origin', 'cost_recorded_at']


def usage_row(event):
    if isinstance(event.get('cost_snapshot'), dict):
        return {k: v for k, v in event['cost_snapshot'].items() if k in FIELDS}
    row = {key: event[key] for key in FIELDS if key in event}
    usage = event.get('usage', {})
    row['cost_status'] = 'usage_unavailable'
    if event['operation'] == 'model_check' and event['status'] == 'ok':
        row.update(estimated_cost_usd='0.000000000000', cost_status='non_generation')
        return row
    if event.get('provider') == 'openai':
        names = {'input_tokens': 'input_tokens', 'output_tokens': 'output_tokens', 'total_tokens': 'total_tokens'}
        row.update({key: usage[value] for key, value in names.items() if value in usage})
        details = usage.get('input_tokens_details') or {}
        row.update(cached_input_tokens=details.get('cached_tokens', 0),
                   cache_write_tokens=details.get('cache_write_tokens', 0),
                   reasoning_tokens=(usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0))
    else:
        names = {'input_tokens': 'promptTokenCount', 'output_tokens': 'candidatesTokenCount',
                 'total_tokens': 'totalTokenCount', 'cached_input_tokens': 'cachedContentTokenCount',
                 'reasoning_tokens': 'thoughtsTokenCount'}
        row.update({key: usage[value] for key, value in names.items() if value in usage})
    if 'input_tokens' not in row or 'output_tokens' not in row:
        return row
    rates = RATES.get((event.get('provider'), event['model']))
    if rates is None:
        row['cost_status'] = 'pricing_unavailable'
        return row
    i, cached, output, write = map(Decimal, rates)
    total_input = row['input_tokens']
    c, w = row.get('cached_input_tokens', 0), row.get('cache_write_tokens', 0)
    if any(type(n) is not int or n < 0 for n in (total_input, c, w, row['output_tokens'])) or c + w > total_input:
        row['cost_status'] = 'invalid_usage'
        return row
    if total_input > 272000:
        i, cached, write, output = i * 2, cached * 2, write * 2, output * Decimal('1.5')
    row.update(input_usd_per_million=str(i), cached_input_usd_per_million=str(cached),
               output_usd_per_million=str(output), cache_write_usd_per_million=str(write),
               pricing_verified='2026-11-09', pricing_source=PRICING_SOURCE)
    # Reasoning tokens are already included in OpenAI output_tokens.
    cost = ((total_input - c - w) * i + c * cached + w * write + row['output_tokens'] * output) / 1000000
    row.update(estimated_cost_usd=format(cost, '.12f'), cost_status='estimated_standard_global')
    return row


class UsageLogger:
    def __init__(self, directory):
        self.store = Store(directory)
        root = self.store.directory.parent if self.store.directory.name in ('self_instruct', 'development_generation', 'benchmark_judging') else self.store.directory
        identity = root / 'api_usage_run.json'
        if not identity.exists():
            atomic_json(identity, {'run_id': str(uuid.uuid4())})
        self.run_id = json.loads(identity.read_text())['run_id']
        self.source = str(self.store.directory.relative_to(root))
        from .usage_cloud import SupabaseLedger
        self.cloud = SupabaseLedger.from_environment()
        if self.cloud:
            self.cloud.check()  # Verify table access before the next paid API call.
        # Freeze historical rows and assign stable IDs once, before any upload.
        events = self.store.read('api_usage.jsonl')
        changed = False
        for event in events:
            if not event.get('event_id'):
                event.update(event_id=str(uuid.uuid4()), run_id=self.run_id, source=self.source)
                changed = True
            if 'cost_snapshot' not in event:
                event['cost_origin'] = 'legacy_reconstructed'
                event['cost_recorded_at'] = now()
                event['cost_snapshot'] = usage_row(event)
                changed = True
        if changed:
            atomic_jsonl(self.store.path('api_usage.jsonl'), events)
        # JSONL is authoritative: repair a CSV interrupted between the two writes.
        path = self.store.path('api_usage.csv')
        temporary = path.with_suffix('.csv.tmp')
        with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for event in self.store.read('api_usage.jsonl'):
                writer.writerow(usage_row(event))
        temporary.replace(path)
        self.sync_pending()

    def sync_pending(self):
        if not self.cloud:
            return 0
        synced = {r['event_id'] for r in self.store.read('api_usage_synced.jsonl')
                  if r.get('ledger_id') == self.cloud.ledger_id}
        pending = [r for r in self.store.read('api_usage.jsonl') if r['event_id'] not in synced]
        for offset in range(0, len(pending), 100):
            batch = pending[offset:offset + 100]
            self.cloud.upload(batch)
            for row in batch:
                self.store.append('api_usage_synced.jsonl', {'event_id': row['event_id'], 'ledger_id': self.cloud.ledger_id})
        return len(pending)

    def __call__(self, event):
        event = {'timestamp': now(), **event, 'event_id': str(uuid.uuid4()),
                 'run_id': self.run_id, 'source': self.source,
                 'cost_origin': 'recorded_at_call', 'cost_recorded_at': now()}
        event['cost_snapshot'] = usage_row(event)
        self.store.append('api_usage.jsonl', event)
        with self.store.path('api_usage.csv').open('a', newline='', encoding='utf-8') as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writerow(usage_row(event))
            stream.flush()
            os.fsync(stream.fileno())
        self.sync_pending()


def export_usage(run_dir):
    """Combine generation folders into run_dir/api_usage.csv; safe to repeat."""
    run_dir = Path(run_dir)
    rows = []
    for journal in sorted(run_dir.rglob('api_usage.jsonl')):
        for event in Store(journal.parent).read(journal.name):
            rows.append({'source': str(journal.parent.relative_to(run_dir)), **usage_row(event)})
    rows.sort(key=lambda row: row.get('timestamp', ''))
    path = run_dir / 'api_usage.csv'
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['source', *FIELDS])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return {'csv': str(path), 'calls': len(rows),
            'estimated_cost_usd': str(sum((Decimal(r['estimated_cost_usd']) for r in rows if 'estimated_cost_usd' in r), Decimal(0))),
            'unknown_cost_calls': sum('estimated_cost_usd' not in r for r in rows)}
