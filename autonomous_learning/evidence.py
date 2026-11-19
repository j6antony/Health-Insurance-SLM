"""Deterministic, task-scoped evidence obligations over final text and source paths."""
from .rendering import parse_text, text_units

EVIDENCE_POLICY = 'minimal-task-evidence-v3'


def resolve_reference(visible, reference):
    parts = reference.split('/')
    if not parts or parts[0] not in ('inventory', 'claim_facts', 'fictional_rule_pack', 'documents'):
        raise ValueError('Unavailable evidence source reference')
    value = visible
    for part in parts:
        if isinstance(value, list):
            if not part.isdigit():
                raise ValueError('Invalid source list index')
            value = value[int(part)]
        else:
            value = value[part]
    return value


def select_evidence(task, visible, findings):
    pages = visible['documents']
    parsed = {p['document_id']: parse_text(p['text']) for p in pages}
    units = {p['document_id']: {u['key']: u['span'] for u in text_units(p)} for p in pages}
    positions = {p['document_id']: n for n, p in enumerate(pages)}
    spans, refs = {}, set()

    def cite(doc, key):
        span = units.get(doc, {}).get(key)
        if span:
            spans[doc, span['start'], span['end']] = span
        else:
            # A missing field has no quote. The complete, verified source text
            # establishes absence, rather than an invented quotation.
            if doc in positions:
                refs.add(f'documents/{positions[doc]}/text')

    def quality(doc):
        if parsed[doc]['readability'] != 'readable':
            cite(doc, 'readability')
        else:
            key = 'signature'
            cite(doc, key)

    def finding(f):
        rule = f['rule_id']
        if rule.startswith(('required.', 'complete.')):
            kind = rule.split('.', 1)[1]
            if kind == 'referral':
                refs.update(('claim_facts/referral_required', 'fictional_rule_pack/conditional'))
                if visible['claim_facts']['referral_required'] is not True:
                    return
            else:
                refs.add('fictional_rule_pack/required/' + visible['claim_category'])
            if rule.startswith('required.') or f['status'] == 'not_evaluated':
                refs.update(('fictional_rule_pack/checks/0', 'fictional_rule_pack/checks/1'))
            else:
                refs.add('fictional_rule_pack/checks/2')
            docs = f['required_evidence']
            if not docs:
                refs.add('inventory')
                return
            for doc in docs:
                cite(doc, 'type')
                if parsed[doc]['readability'] != 'readable':
                    cite(doc, 'readability')
                elif rule.startswith('complete.'):
                    fields = parsed[doc]['fields']
                    missing = [key for key in ('patient_id', 'service_date') if not fields.get(key)]
                    if parsed[doc]['signature'] is False:
                        cite(doc, 'signature')
                    elif missing:
                        refs.add(f'documents/{positions[doc]}/text')
                    elif fields['patient_id'] != visible['claim_facts']['patient_id']:
                        cite(doc, 'patient_id')
                        refs.add('claim_facts/patient_id')
                    elif parsed[doc]['signature'] is None:
                        cite(doc, 'signature')
                    else:
                        for key in ('patient_id', 'service_date', 'signature'):
                            cite(doc, key)
                        refs.add('claim_facts/patient_id')
                else:
                    quality(doc)
        else:
            refs.add('fictional_rule_pack/checks/3')
            dated = [(doc, p['fields'].get('service_date')) for doc, p in parsed.items() if p['readability'] == 'readable']
            conflict = next(((a, b) for a, av in dated for b, bv in dated if av and bv and av != bv), None)
            if conflict:
                for doc in conflict:
                    cite(doc, 'service_date')
            else:
                refs.add('inventory')
                if f['status'] == 'requires_review':
                    for doc, document in parsed.items():
                        if document['readability'] != 'readable':
                            cite(doc, 'readability')
                for doc, value in dated:
                    cite(doc, 'service_date')
                    if not value:
                        cite(doc, 'type')

    if task == 'document_classification':
        for doc in parsed:
            cite(doc, 'type')
    elif task == 'field_extraction':
        for doc, p in parsed.items():
            if p['readability'] != 'readable':
                cite(doc, 'readability')
            for key in p['fields']:
                cite(doc, key)
    elif task == 'evidence_quality':
        for doc in parsed:
            quality(doc)
    elif task == 'missing_documents':
        # A typed verified inventory proves presence/absence without invented
        # quotes. Legacy untyped inventories need the identifying headings too.
        for item in visible['inventory']:
            if item['present'] and item['document_type'] is None:
                cite(item['document_id'], 'type')
        refs.update(('inventory', 'fictional_rule_pack/required/' + visible['claim_category'],
                     'claim_facts/referral_required', 'fictional_rule_pack/conditional'))
    else:
        selected = [f for f in findings if task == 'finding_explanation'
                    or task == 'conditional_requirements' and f['rule_id'] == 'required.referral'
                    or task == 'cross_document_consistency' and f['rule_id'].startswith('consistency.')]
        for f in selected:
            finding(f)
    for ref in refs:
        resolve_reference(visible, ref)
    return list(spans.values()), sorted(refs)


def check_offsets(payload, answer):
    docs = {(p['document_id'], p['page_number']): p['text'] for p in payload['scenario']['documents']}
    seen = set()
    for e in answer['evidence']:
        key = (e['document_id'], e['page_number'], e['start'], e['end'])
        if key in seen or e['end'] > len(docs.get(key[:2], '')) or e['start'] < 0 or e['end'] <= e['start'] or not e['quote'] or docs.get(key[:2], '')[e['start']:e['end']] != e['quote']:
            raise ValueError('Invalid or duplicate evidence offset')
        seen.add(key)
    if len(answer['source_references']) != len(set(answer['source_references'])):
        raise ValueError('Duplicate source reference')
    for ref in answer['source_references']:
        resolve_reference(payload['scenario'], ref)


def assess_evidence(case, answer, reference_reviewed=False):
    """Exact deterministic proof obligations; reviewed anchors for unfamiliar prose."""
    check_offsets(case['payload'], answer)
    from .tasks import target_from_visible
    try:
        expected = target_from_visible(case)
    except (ValueError, KeyError, TypeError):
        if not reference_reviewed:
            return {'available': False, 'relevant': False, 'sufficient': False, 'basis': 'semantic_grounding_unavailable'}
        expected = case['target']
        basis = 'independently_reviewed_reference_anchors'
    else:
        basis = 'deterministic_task_obligations'
    encode = lambda span: (span['document_id'], span['page_number'], span['start'], span['end'], span['quote'])
    required, cited = set(map(encode, expected['evidence'])), set(map(encode, answer['evidence']))
    required_refs, cited_refs = set(expected['source_references']), set(answer['source_references'])
    return {'available': True, 'relevant': cited <= required and cited_refs <= required_refs,
            'sufficient': required <= cited and required_refs <= cited_refs, 'basis': basis}
