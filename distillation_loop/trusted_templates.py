"""Executable fictional-rule fixtures for independent arithmetic/date/logic training.

These are narrow synthetic skills, not independent real-world evaluation data.
Source and target are constructed by code; no teacher approval is involved.
"""
from datetime import date, timedelta
from decimal import Decimal
from self_instruct_pipeline.deduplicator import Deduplicator
from self_instruct_pipeline.validation import validate_answer

TEMPLATE_VERSION = 'fictional-rules-v1'


def template_pair(index):
    family = index % 3
    case = index // 3
    if family == 0:
        task = 'patient_billing'
        billed = Decimal(100 + case * 17) + Decimal('0.35')
        paid = Decimal((case % 4) * 25)
        adjusted = Decimal((case % 3) * 10) + Decimal('0.10')
        if case % 6 == 0:
            paid, adjusted = Decimal(0), Decimal(0)
        elif case % 6 == 1:
            paid, adjusted = billed, Decimal(0)  # Exactly settled.
        elif case % 6 == 2:
            paid, adjusted = billed + Decimal(25), Decimal(0)  # Credit balance.
        elif case % 6 == 3:
            paid, adjusted = Decimal(0), billed  # Fully adjusted.
        elif case % 6 == 4:
            billed = paid = adjusted = Decimal(0)
        result = billed - paid - adjusted
        source = (f'Fictional training ledger {case}. Amounts in training USD. Billed: {billed:.2f}; '
                  f'paid: {paid:.2f}; adjustment: {adjusted:.2f}. '
                  'Training rule: remaining balance = billed minus paid minus adjustment. '
                  'This ledger supplies no real coverage or patient-liability determination.')
        instruction = 'Compute the remaining training-ledger balance only. Do not infer patient liability. Report the amount in USD.'
        answer = f'The remaining training-ledger balance is USD {result:.2f}. Patient liability is not determined by this calculation.'
        sign = "negative" if result < 0 else "zero" if result == 0 else "positive"
        variant = f'paid_zero={paid == 0};adjustment_zero={adjusted == 0};balance={sign}'
        context = {'ledger': {'billed': str(billed), 'paid': str(paid), 'adjustment': str(adjusted)}}
    elif family == 1:
        task = 'appeals_process_qa'
        month, day = ((1, 31), (2, 28), (12, 31), (4, 30), (11, 30))[case % 5]
        start = date(2024 + case % 3, month, day)
        offset = (0, 1, 7, 30, 60)[case % 5]
        result = (start + timedelta(days=offset)).isoformat()
        source = (f'Fictional exercise rule {case}: start date {start.isoformat()}. '
                  f'The exercise deadline is {offset} calendar days after the start date; '
                  'the start date is day zero. Weekends and holidays count. No business-day adjustment. '
                  'This is not an actual insurer appeal deadline.')
        instruction = 'Calculate the fictional exercise deadline in ISO date format using only the supplied counting rule.'
        answer = f'The fictional exercise deadline is {result}.'
        variant = f'offset={offset};month_crossing={start.month != (start + timedelta(days=offset)).month}'
        context = {'counting_rule': 'calendar days; start date is day zero', 'start_date': start.isoformat(), 'offset_days': offset}
    else:
        task = 'medical_necessity_support'
        first, second = bool(case % 2), bool((case // 2) % 2)
        conjunction = 'AND' if (case // 4) % 2 == 0 else 'OR'
        result = (first and second) if conjunction == 'AND' else (first or second)
        source = (f'Fictional training criteria {case}: criterion A is {str(first).lower()}; '
                  f'criterion B is {str(second).lower()}. The exercise criteria are met if A {conjunction} B. '
                  'Both values are explicitly established. These fictional criteria do not establish '
                  'medical necessity, coverage, or actual authorization.')
        instruction = 'Evaluate only whether the fictional exercise criteria are met. Do not make a clinical or coverage determination.'
        answer = f'The fictional exercise criteria are {"met" if result else "not met"}. No clinical or coverage determination is established.'
        variant = f'{conjunction}:A={first};B={second}'
        context = {'criterion_A': first, 'criterion_B': second, 'relationship': conjunction}
    # Pin the code-computed answer and review flags through the existing contract.
    response = {'answer': answer, 'evidence': [{'document_id': 'RULE', 'quote': source}],
                'missing_information': [], 'requires_review': False}
    if task == 'medical_necessity_support':
        response['criteria_assessment'] = [f'A={str(first).lower()}; B={str(second).lower()}; relationship={conjunction}; met={str(result).lower()}.']
    record = {'id': f'{TEMPLATE_VERSION}-{index}', 'task': task, 'instruction': instruction,
              'scenario': {'synthetic': True, 'documents': [{'document_id': 'RULE', 'page_number': 1, 'text': source}], 'context': context},
              'case_type': 'conditional' if family == 2 else 'supported', 'difficulty': 'medium',
              'verification_basis': TEMPLATE_VERSION, 'template_family': task, 'reasoning_variant': variant,
              'answer_checks': [{'path': [key], 'operation': 'equals', 'operands': [value]}
                                for key, value in response.items() if key != 'evidence']}
    return record, response


def template_pairs(count, schemas, exclusions):
    # Bound scanning if held-out data happen to exclude a complete template family.
    dedup = Deduplicator(exclusions)
    pairs = []
    for index in range(max(100, count * 20)):
        if len(pairs) == count:
            break
        record, answer = template_pair(index)
        if dedup.check(record):
            continue
        validate_answer(record, answer, schemas[record['task']])
        pairs.append((record, answer))
    if len(pairs) != count:
        raise ValueError('Trusted template coverage exhausted by held-out exclusions')
    return pairs
