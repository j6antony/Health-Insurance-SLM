#!/usr/bin/env python3
"""Checks the consolidated dataset. Run from any working directory."""
from pathlib import Path
import json, collections, sys
root=Path(__file__).resolve().parents[1]
errors=[]
def load(name):
    p=root/'data'/name
    if not p.is_file():
        errors.append(f'Missing {p}'); return []
    with p.open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]

expected={'noisy_images':(700,150,150),'harvard':(140,30,30),'mtsamples':(3476,745,745),'benchmark':(99,33,33)}
for prefix,counts in expected.items():
    parts=[]
    for sp,expected_count in zip(('train','validation','test'),counts):
        rows=load(f'splits/{prefix}_{sp}.jsonl')
        if len(rows)!=expected_count: errors.append(f'{prefix} {sp}: expected {expected_count}, got {len(rows)}')
        parts.append(rows)
    key='patient_group_id' if prefix=='benchmark' else ('group_id' if prefix!='harvard' else 'group_id')
    group_sets=[set(x.get(key) for x in rows) for rows in parts]
    if any(group_sets[i]&group_sets[j] for i,j in ((0,1),(0,2),(1,2))):
        errors.append(f'{prefix}: group leakage')
    if prefix=='noisy_images':
        for rows in parts:
            for row in rows:
                p=root/row.get('image_path','')
                if not p.is_file():errors.append('Missing image '+str(p))
    if prefix=='benchmark':
        for rows in parts:
            for row in rows:
                pr=row.get('provenance',{})
                p=root/pr.get('file','')
                if not p.is_file():errors.append('Missing PDF '+str(p))
    print(f'{prefix}: {list(map(len,parts))} records; group isolation checked')
for f,n in [('splits/verification/train_cases.jsonl',16),('splits/verification/validation_cases.jsonl',4),('splits/verification/test_cases.jsonl',4)]:
    if len(load(f))!=n:errors.append(f'{f}: incorrect record count')
print('ERRORS:',len(errors))
for e in errors[:20]:print(' -',e)
if errors: sys.exit(1)
print('PASS: dataset references and split counts verified')
