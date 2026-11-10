"""Optional independently authored answer assertions; never passed to student prompts.

Only attach these after reviewing source applicability. Assertions authored by the
teacher itself do not establish independent truth.
"""
from datetime import date, timedelta
from decimal import Decimal


def check_rules(record, answer):
    for rule in record.get('answer_checks', []):
        actual = answer
        for part in rule['path']:
            actual = actual[part]
        op = rule['operation']
        args = rule['operands']
        if op == 'equals':
            expected = args[0]
        elif op in ('sum', 'subtract'):
            values = [Decimal(str(v)) for v in args]
            expected = sum(values) if op == 'sum' else values[0] - sum(values[1:])
            actual = Decimal(str(actual))
        elif op == 'all':
            if any(type(v) is not bool for v in args):
                raise ValueError('Boolean rule operands required')
            expected = all(args)
        elif op == 'any':
            if any(type(v) is not bool for v in args):
                raise ValueError('Boolean rule operands required')
            expected = any(args)
        elif op == 'date_offset':
            expected = (date.fromisoformat(args[0]) + timedelta(days=int(args[1]))).isoformat()
        else:
            raise ValueError('Unsupported answer check operation')
        if type(expected) is bool and type(actual) is not bool:
            raise ValueError('Boolean answer required')
        if actual != expected:
            raise ValueError(f'Independent answer check failed at {rule["path"]}')
