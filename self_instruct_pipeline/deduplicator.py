import json
import re
from collections import Counter
from .persistence import digest


def tokens(text):
    return re.findall(r'\w+', text.casefold())


def rouge_l(left, right):
    a, b = tokens(left), tokens(right)
    if not a or not b:
        return 0.0
    # Rolling-row LCS: same ROUGE-L F1 idea as the original, without its API stack.
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b):
            current.append(previous[j] + 1 if x == y else max(previous[j + 1], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(a) + len(b))


def canonical_scenario(row):
    if 'scenario' in row:
        return row['scenario']
    value = row.get('input')
    if value is None:
        value = json.loads(row['instances'][0]['input'].split('\nResponse schema:\n', 1)[0])
    return value


def fingerprint(row):
    return digest({'task': row['task'], 'instruction': ' '.join(tokens(row['instruction'])),
                   'scenario': canonical_scenario(row)})


def reasoning_signature(text):
    # Preserve explicit counterfactuals rather than dropping them for superficial
    # similarity. These are guardrails, not semantic entailment detection.
    patterns = [r'\bunreadable\b|\bocr failed\b', r'\bmissing\b|\babsent\b',
                r'\bor\b', r'\band\b', r'\bunknown\b|\bunspecified\b',
                r'\bwill\b|\bplanned\b|\bintends\b', r'\bcompleted\b|\breceived\b',
                r'\bexcept\b|\bexempt\b', r'\bconflict\w*\b', r'\bnot\b|\bno\b']
    return tuple(bool(re.search(pattern, text, re.I)) for pattern in patterns)


class Deduplicator:
    def __init__(self, rows, instruction_threshold=0.9, context_threshold=0.88, semantic_threshold=0.85):
        self.rows = []
        self.fingerprints = set()
        self.instruction_threshold = instruction_threshold
        self.context_threshold = context_threshold
        self.semantic_threshold = semantic_threshold
        for row in rows:
            self.add(row)

    def add(self, row):
        self.fingerprints.add(fingerprint(row))
        self.rows.append(row)

    def check(self, row):
        if fingerprint(row) in self.fingerprints:
            return 'exact_duplicate'
        scenario = json.dumps(canonical_scenario(row), sort_keys=True)
        signature = reasoning_signature(scenario)
        for old in self.rows:
            if old['task'] != row['task']:
                continue
            old_context = json.dumps(canonical_scenario(old), sort_keys=True)
            if signature != reasoning_signature(old_context):
                continue
            # Measure context and instruction together: repeated instruction
            # templates alone are expected and must not eliminate valid cases.
            a, b = set(tokens(scenario)), set(tokens(old_context))
            similarity = len(a & b) / max(1, len(a | b))
            if similarity < min(self.context_threshold, self.semantic_threshold):
                continue
            if rouge_l(row['instruction'], old['instruction']) >= self.instruction_threshold and similarity >= self.context_threshold:
                return 'near_duplicate_instruction_and_context'
        return None

    def neighbors(self, row, limit=3):
        """Shortlist same-task examples for the separate semantic novelty audit."""
        a = set(tokens(json.dumps(canonical_scenario(row))))
        scored = []
        for old in self.rows:
            if old['task'] != row['task']:
                continue
            b = set(tokens(json.dumps(canonical_scenario(old))))
            score = len(a & b) / max(1, len(a | b))
            scored.append((score, old))
        scored.sort(key=lambda item: item[0], reverse=True)
        # Always include the nearest example; include others above the configured
        # shortlist threshold. The model audit, not lexical overlap, judges meaning.
        return [{'id': r.get('id', r.get('generation_id')), 'task': r['task'],
                 'instruction': r['instruction'], 'scenario': canonical_scenario(r)}
                for i, (score, r) in enumerate(scored[:limit]) if i == 0 or score >= self.semantic_threshold]
