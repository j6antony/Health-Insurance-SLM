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


def structured_template_pair(index):
    task = ('document_classification', 'field_extraction', 'claim_scrubbing')[index % 3]
    case = index // 3
    if task == 'document_classification':
        heading, label = (
            ('Itemized hospital bill', 'hospital_bill'),
            ('Discharge summary', 'discharge_summary'),
            ('Medication prescription', 'prescription'),
            ('Laboratory investigation report', 'investigation_report'),
            ('Insurance claim form', 'claim_form'),
        )[case % 5]
        source = f'{heading} | Synthetic record {case}'
        instruction = 'Classify this document using the heading and the supplied taxonomy.'
        answer = {'document_type': label, 'evidence': [heading], 'requires_review': False}
        checked = {'document_type': label, 'requires_review': False}
    elif task == 'field_extraction':
        identifier = f'POL-{case:05d}'
        source = f'Policy number: {identifier}. Synthetic intake record {case}.'
        instruction = 'Extract the policy number only and cite its source.'
        answer = {'fields': [{'field_name': 'policy_number', 'value': identifier,
                             'evidence': identifier, 'page_number': 1, 'requires_review': False}]}
        checked = {}
    else:
        signature, date = bool(case & 1), bool(case & 2)
        complete = signature and date
        source = (f'Synthetic intake checklist {case}. Signature present: {signature}. '
                  f'Service date present: {date}. Review is required unless both are present.')
        missing = [name for name, present in [('Signature', signature), ('Service date', date)] if not present]
        instruction = 'Apply the supplied checklist and identify any missing items.'
        answer = {'answer': 'Checklist complete.' if complete else 'Checklist incomplete.',
                  'missing_information': missing, 'issues': missing,
                  'requires_review': not complete, 'evidence': [{'document_id': 'RULE', 'quote': source}]}
        checked = {'missing_information': missing, 'issues': missing}
    rules = [{'path': [key], 'operation': 'equals', 'operands': [value]} for key, value in checked.items()]
    if task == 'claim_scrubbing':
        rules.append({'path': ['requires_review'], 'operation': 'any', 'operands': [not signature, not date]})
    if task == 'field_extraction':
        rules.extend({'path': ['fields', 0, key], 'operation': 'equals', 'operands': [value]}
                     for key, value in answer['fields'][0].items() if key != 'evidence')
    record = {'id': f'fictional-structured-v1-{index}', 'task': task, 'instruction': instruction,
              'scenario': {'synthetic': True, 'documents': [{'document_id': 'RULE', 'page_number': 1, 'text': source}],
                           'context': {}},
              'case_type': 'conditional', 'difficulty': 'medium', 'verification_basis': 'fictional-structured-v1',
              'template_family': task, 'reasoning_variant': str(case % 5), 'answer_checks': rules}
    return record, answer


def contextual_template_pair(index):
    record, answer = template_pair(index)
    task = record['task']
    context = record['scenario']['context']
    case = index // 3
    if task == 'patient_billing':
        ledger = context['ledger']
        documents = [
            ('PROVIDER-STATEMENT', f'Patient accounts | Encounter SYN-{case}\nTotal charges: USD {ledger["billed"]}.'),
            ('REMITTANCE', f'Remittance for SYN-{case}\nPlan payment: USD {ledger["paid"]}. Contractual adjustment: USD {ledger["adjustment"]}.'),
            ('LEDGER-GUIDE', 'For this ledger, subtract the plan payment and contractual adjustment from total charges. The result is an accounting balance only.'),
        ]
        record['instruction'] = 'Reconcile the provider statement and remittance. Report the accounting balance in USD.'
    elif task == 'appeals_process_qa':
        documents = [
            ('NOTICE', f'Appeals correspondence | File SYN-{case}\nNotice date: {context["start_date"]}.'),
            ('PLAN-ADDENDUM', f'For this notice, the submission window closes {context["offset_days"]} calendar days after the notice date. The notice date is day zero. Weekends and holidays count.'),
        ]
        record['instruction'] = 'Calculate the submission deadline from the notice and the supplied plan addendum. Use ISO date format.'
        answer['answer'] = answer['answer'].replace('fictional exercise deadline', 'submission deadline')
    else:
        weeks = 8 if context['criterion_A'] else 4
        authorization = 'attached and signed' if context['criterion_B'] else 'not supplied'
        relationship = 'both requirements' if context['relationship'] == 'AND' else 'at least one requirement'
        documents = [
            ('PLAN-EXCERPT', f'For this sample plan’s documentation checklist, {relationship} must be satisfied. The requirements are six completed therapy weeks and a signed referral.'),
            ('THERAPY-SUMMARY', f'Rehabilitation progress summary | File SYN-{case}\nCompleted therapy: {weeks} weeks.'),
            ('REFERRAL-LOG', f'Intake referral log | File SYN-{case}\nReferral status: {authorization}.'),
        ]
        record['instruction'] = 'Apply the documentation checklist to the submitted records. State whether the checklist is satisfied and explain each requirement.'
        met = context['criterion_A'] and context['criterion_B'] if context['relationship'] == 'AND' else context['criterion_A'] or context['criterion_B']
        answer['answer'] = f'The documentation checklist is {"satisfied" if met else "not satisfied"}.'
        answer['criteria_assessment'] = [f'Completed therapy: {weeks} weeks. Signed referral: {authorization}.']
    record['id'] = f'contextual-rules-v1-{index}'
    record['verification_basis'] = 'contextual-rules-v1'
    record['scenario'] = {'synthetic': True, 'context': {}, 'documents': [
        {'document_id': name, 'page_number': 1, 'text': text} for name, text in documents]}
    answer['evidence'] = [{'document_id': name, 'quote': text.split('\n')[-1]} for name, text in documents]
    for rule in record['answer_checks']:
        rule['operands'] = [answer[rule['path'][0]]]
    return record, answer


def template_pairs(count, schemas, exclusions, *, expanded=False):
    # Bound scanning if held-out data happen to exclude a complete template family.
    dedup = Deduplicator(exclusions)
    pairs = []
    for index in range(max(100, count * 20)):
        if len(pairs) == count:
            break
        if expanded:
            record, answer = (contextual_template_pair if index % 2 == 0 else structured_template_pair)(index // 2)
        else:
            record, answer = template_pair(index)
        if dedup.check(record):
            continue
        validate_answer(record, answer, schemas[record['task']])
        pairs.append((record, answer))
    if len(pairs) != count:
        raise ValueError('Trusted template coverage exhausted by held-out exclusions')
    return pairs
