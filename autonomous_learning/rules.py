"""Fictional document checklist, never coverage or medical adjudication."""
from datetime import date
from decimal import Decimal

VERSION = 'fictional-document-rules-v1'
STATUSES = ('passed', 'issue_detected', 'requires_review', 'not_applicable', 'not_evaluated')
CATEGORIES = ('hospitalization', 'outpatient', 'diagnostic', 'pharmacy')
REQUIRED = {
    'hospitalization': ('claim_form', 'hospital_bill', 'discharge_summary'),
    'outpatient': ('claim_form', 'invoice'),
    'diagnostic': ('claim_form', 'investigation_report'),
    'pharmacy': ('claim_form', 'pharmacy_invoice', 'prescription'),
}
HEADINGS = {'claim_form': 'Claim form', 'hospital_bill': 'Itemized hospital bill',
            'discharge_summary': 'Discharge summary', 'invoice': 'Outpatient invoice',
            'investigation_report': 'Investigation report', 'pharmacy_invoice': 'Pharmacy invoice',
            'prescription': 'Medication prescription', 'referral': 'Signed referral'}


def rule_pack():
    return {'version': VERSION, 'fictional': True, 'scope': 'Document verification only',
            'required': {k: list(v) for k, v in REQUIRED.items()}, 'conditional': 'A referral is required when referral_required is true.',
            'checks': ['Presence', 'Readability', 'patient_id, service_date and signature',
                       'Service dates agree across readable documents'], 'headings': HEADINGS}


def validate_state(s):
    if s['category'] not in CATEGORIES or s['rule_pack_version'] != VERSION or s.get('synthetic') is not True:
        raise ValueError('Unsupported category, version or non-synthetic source')
    if type(s['facts']['referral_required']) not in (bool, type(None)):
        raise ValueError('Conditional applicability must be true, false or unknown')
    documents = s['documents']
    if len({d['id'] for d in documents}) != len(documents) or len({d['type'] for d in documents}) != len(documents):
        raise ValueError('Duplicate document identity/type')
    for d in documents:
        if d['type'] not in HEADINGS or type(d['present']) is not bool:
            raise ValueError('Unknown document or invalid presence')
        if d['readability'] not in ('readable', 'unreadable', 'unknown', 'not_available'):
            raise ValueError('Unknown evidence readability')
        if not d['present']:
            if d['readability'] != 'not_available' or d['fields'] or d['signature'] is not None:
                raise ValueError('Missing document cannot contain evidence')
        else:
            if d['readability'] == 'not_available' or type(d['signature']) not in (bool, type(None)):
                raise ValueError('Infeasible present document state')
            for field in ('service_date',):
                if d['fields'].get(field):
                    date.fromisoformat(d['fields'][field])
            if 'charges' in d['fields']:
                charges = [Decimal(v) for v in d['fields']['charges']]
                total = Decimal(d['fields']['total'])
                if any(not v.is_finite() or v < 0 for v in [*charges, total]) or sum(charges) != total:
                    raise ValueError('Invalid charge arithmetic')
    return s


def evaluate(s):
    validate_state(s)
    findings = []
    by_type = {d['type']: d for d in s['documents']}

    def finding(rule, status, reason, facts, predicates, refs):
        findings.append({'rule_id': rule, 'rule_pack_version': VERSION, 'input_facts': facts,
                         'predicates': predicates, 'status': status, 'reason': reason,
                         'required_evidence': refs, 'requires_review': status in ('requires_review', 'issue_detected')})

    types = list(REQUIRED[s['category']]) + ['referral']
    for kind in dict.fromkeys(types):
        condition = s['facts']['referral_required'] if kind == 'referral' else True
        d = by_type.get(kind)
        refs = [d['id']] if d and d['present'] else []
        present = bool(d and d['present'])
        rule = 'required.' + kind
        if condition is False:
            status, reason = 'not_applicable', 'The fictional conditional requirement does not apply.'
        elif condition is None:
            status, reason = 'requires_review', 'Referral applicability is unknown.'
        elif not present:
            status, reason = 'issue_detected', 'Required document is absent from the supplied inventory.'
        elif d['readability'] != 'readable':
            status, reason = 'requires_review', 'Document is present but its contents cannot be verified.'
        else:
            status, reason = 'passed', 'Required document is present and readable.'
        finding(rule, status, reason, {'document_type': kind, 'applicable': condition, 'present': present},
                {'applicable': condition, 'present': present}, refs)
        if not present or condition is not True or d['readability'] != 'readable':
            status, reason = 'not_evaluated', 'Content checks were not evaluated without applicable readable evidence.'
        else:
            fields = d['fields']
            if d['signature'] is False or any(not fields.get(k) for k in ('patient_id', 'service_date')):
                status, reason = 'issue_detected', 'Required signature or field is incomplete.'
            elif fields['patient_id'] != s['facts']['patient_id']:
                status, reason = 'issue_detected', 'Patient identifiers conflict across supplied records.'
            elif d['signature'] is None:
                status, reason = 'requires_review', 'Signature evidence is unknown.'
            else:
                status, reason = 'passed', 'Required fields and signature are present and consistent.'
        finding('complete.' + kind, status, reason,
                {'document_type': kind, 'fields': d['fields'] if d else {}, 'signature': d['signature'] if d else None},
                {'content_evaluated': status != 'not_evaluated'}, refs)
    readable = [d for d in s['documents'] if d['present'] and d['readability'] == 'readable']
    dates = {d['fields'].get('service_date') for d in readable if d['fields'].get('service_date')}
    if len(dates) > 1:
        status, reason = 'issue_detected', 'Readable documents contain conflicting service dates.'
    elif len(readable) < 2 or any(not d['fields'].get('service_date') for d in readable):
        status, reason = 'requires_review', 'Insufficient date evidence for cross-document verification.'
    else:
        status, reason = 'passed', 'Readable service dates agree.'
    finding('consistency.service_dates', status, reason, {'dates': sorted(dates)},
            {'multiple_dates': len(dates) > 1}, [d['id'] for d in readable])
    return findings
