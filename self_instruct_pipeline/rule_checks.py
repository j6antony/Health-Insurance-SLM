"""Strict independently authored answer assertions, excluded from target prompts."""
from datetime import date, timedelta
from decimal import Decimal


def _number(value):
    if type(value) not in (str, int, float):
        raise ValueError('Numeric operand required')
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError('Finite numeric operand required')
    return number


def check_rules(record, answer):
    rules = record.get('answer_checks', [])
    if not isinstance(rules, list):
        raise ValueError('answer_checks must be an array')
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {'path', 'operation', 'operands'}:
            raise ValueError('Answer check requires exactly path, operation, operands')
        path, args = rule['path'], rule['operands']
        if not isinstance(path, list) or not path or not isinstance(args, list) or not args:
            raise ValueError('Nonempty check path and operands required')
        actual = answer
        for part in path:
            if isinstance(actual, list):
                if type(part) is not int or part < 0:
                    raise ValueError('Nonnegative array index required')
            elif not isinstance(actual, dict) or not isinstance(part, str):
                raise ValueError('Object key required')
            actual = actual[part]
        op = rule['operation']
        if op == 'equals':
            if len(args) != 1:
                raise ValueError('equals requires one operand')
            expected = args[0]
            if type(actual) is not type(expected):
                raise ValueError('Exact check requires matching types')
        elif op in ('sum', 'subtract'):
            values = [_number(v) for v in args]
            expected = sum(values) if op == 'sum' else values[0] - sum(values[1:])
            actual = _number(actual)
        elif op in ('all', 'any'):
            if any(type(v) is not bool for v in args) or type(actual) is not bool:
                raise ValueError('Boolean rule operands and answer required')
            expected = all(args) if op == 'all' else any(args)
        elif op == 'date_offset':
            if len(args) != 2 or type(args[1]) is not int or not isinstance(args[0], str):
                raise ValueError('date_offset requires ISO date and integer calendar-day offset')
            expected = (date.fromisoformat(args[0]) + timedelta(days=args[1])).isoformat()
        else:
            raise ValueError('Unsupported answer check operation')
        if actual != expected:
            raise ValueError(f'Independent answer check failed at {path}')
