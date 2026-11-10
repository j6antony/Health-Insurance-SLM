"""Optional Supabase ledger. Only usage metadata is uploaded, never prompts or keys."""
import csv
import hashlib
import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import UUID


class UsageSyncError(RuntimeError):
    pass


class SupabaseLedger:
    def __init__(self, url, key, opener=None):
        parsed = urllib.parse.urlsplit(url.rstrip('/'))
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.path not in ('', '/') or parsed.query or parsed.fragment:
            raise ValueError('SUPABASE_URL must be an HTTPS project origin')
        if not key:
            raise ValueError('SUPABASE_SECRET_KEY is empty')
        self.base, self.key = url.rstrip('/') + '/rest/v1/api_usage_events', key
        self.ledger_id = hashlib.sha256(self.base.encode()).hexdigest()
        context = ssl.create_default_context()
        if not ssl.get_default_verify_paths().cafile and Path('/etc/ssl/cert.pem').is_file():
            context = ssl.create_default_context(cafile='/etc/ssl/cert.pem')
        self.opener = opener or (lambda request, timeout: urllib.request.urlopen(request, timeout=timeout, context=context))

    @classmethod
    def from_environment(cls):
        url, key = os.environ.get('SUPABASE_URL'), os.environ.get('SUPABASE_SECRET_KEY')
        if not url and not key:
            return None
        if not url or not key:
            raise UsageSyncError('Set both SUPABASE_URL and SUPABASE_SECRET_KEY, or neither for local-only logging')
        return cls(url, key)

    def request(self, query, payload=None):
        headers = {'apikey': self.key, 'Content-Type': 'application/json'}
        # New sb_secret keys belong only in apikey. Legacy service-role JWTs
        # also need the Authorization header for PostgREST role selection.
        if not self.key.startswith('sb_secret_'):
            headers['Authorization'] = 'Bearer ' + self.key
        if payload is not None:
            headers['Prefer'] = 'resolution=ignore-duplicates,return=minimal'
        request = urllib.request.Request(self.base + '?' + urllib.parse.urlencode(query),
            data=json.dumps(payload).encode() if payload is not None else None,
            headers=headers, method='POST' if payload is not None else 'GET')
        try:
            with self.opener(request, timeout=20) as response:
                raw = response.read()
                return json.loads(raw) if raw else []
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            raise UsageSyncError(f'Supabase ledger HTTP {status}; local usage is retained. Fix sync before resuming.') from None
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            raise UsageSyncError('Supabase ledger unavailable or invalid response; local usage is retained. Fix sync before resuming.') from None

    def check(self):
        self.request({'select': 'event_id', 'limit': 1})

    def upload(self, events):
        if not events:
            return
        from .usage import usage_row
        payload = []
        for event in events:
            row = usage_row(event)
            payload.append({
                'event_id': event['event_id'], 'run_id': event['run_id'], 'source': event.get('source', '.'),
                'logged_at': event.get('timestamp'), 'provider': row.get('provider'),
                'model': row.get('model'), 'operation': row.get('operation'), 'status': row.get('status'),
                'input_tokens': row.get('input_tokens'), 'output_tokens': row.get('output_tokens'),
                'total_tokens': row.get('total_tokens'), 'estimated_cost_usd': row.get('estimated_cost_usd'),
                'details': row})
        self.request({'on_conflict': 'event_id'}, payload)

    def export_csv(self, output_path, run_id=None):
        from .usage import FIELDS
        from decimal import Decimal
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        offset, total, unknown, cost = 0, 0, 0, Decimal(0)
        with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=['source', *FIELDS])
            writer.writeheader()
            while True:
                query = {'select': 'source,details', 'order': 'created_at.asc,event_id.asc', 'limit': 500, 'offset': offset}
                if run_id:
                    query['run_id'] = 'eq.' + str(UUID(str(run_id)))
                rows = self.request(query)
                if not isinstance(rows, list):
                    raise UsageSyncError('Supabase export returned an invalid page')
                if not rows:
                    break
                for row in rows:
                    details = {k: v for k, v in row['details'].items() if k in FIELDS}
                    writer.writerow({'source': row['source'], **details})
                    value = details.get('estimated_cost_usd')
                    if value is None:
                        unknown += 1
                    else:
                        cost += Decimal(str(value))
                offset += len(rows)
                total += len(rows)
        temporary.replace(path)
        return {'csv': str(path), 'calls': total, 'estimated_cost_usd': str(cost), 'unknown_cost_calls': unknown}


def export_cloud_usage(output_path, run_id=None):
    ledger = SupabaseLedger.from_environment()
    if ledger is None:
        raise UsageSyncError('Set SUPABASE_URL and SUPABASE_SECRET_KEY to export cloud history')
    return ledger.export_csv(output_path, run_id)


def sync_usage_history(run_dir):
    """Upload existing local journals, including calls from before cloud was enabled."""
    from .usage import UsageLogger
    if SupabaseLedger.from_environment() is None:
        raise UsageSyncError('Set both Supabase credentials before syncing usage history')
    count = 0
    for path in sorted(Path(run_dir).rglob('api_usage.jsonl')):
        UsageLogger(path.parent)  # Migrates stable IDs/snapshots, then syncs pending rows.
        count += 1
    return {'journals_synced': count}
