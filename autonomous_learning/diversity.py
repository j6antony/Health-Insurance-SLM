"""Observed surface forms, independent of patient IDs and numeric fact values."""
from collections import defaultdict
import re
from .rendering import parse_text
from .storage import digest


def surface_signature(page):
    text = re.sub(r'SYN-[PC]-[a-f0-9]+', '<ID>', page['text'])
    text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', '<MONTH>', text)
    text = re.sub(r'\d+', '<N>', text)
    return digest(text)


def measure(pages):
    signatures, examples = defaultdict(set), defaultdict(set)
    for page in pages:
        kind = parse_text(page['text'])['type']
        signatures[kind].add(surface_signature(page))
        examples[kind].add(page['text'])
    return {'distinct_surface_forms_by_type': {k: len(v) for k, v in signatures.items()},
            'unique_document_texts_by_type': {k: len(v) for k, v in examples.items()},
            'basis': 'Accepted final text after replacing identifiers and numbers; no model diversity score inferred.'}
