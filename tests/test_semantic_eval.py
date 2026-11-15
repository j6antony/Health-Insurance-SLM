import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from distillation_loop.benchmarks import fixed_family_split
from distillation_loop.colab import distillation_1000_configs
from distillation_loop.core import load_loop_config
from distillation_loop.semantic_eval import SCHEMAS, run
from self_instruct_pipeline.config import load_config
from self_instruct_pipeline.persistence import atomic_json, atomic_jsonl, digest


class SemanticReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schemas = json.loads(SCHEMAS.read_text())['tasks']
        _, config = distillation_1000_configs(load_config(), load_loop_config(), 'unused')
        config['development_examples_per_task'] = 20
        config['evaluation'].update(holdout_families_per_task=2, total_examples=360)
        cls.development, cls.holdout = fixed_family_split(config, cls.schemas)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.split = self.root / 'split.json'
        self.output = self.root / 'output'
        atomic_json(self.split, {
            name: {'case_ids': [x['record']['id'] for x in rows], 'sha256': digest(rows),
                   'source_groups': sorted({x['source_group'] for x in rows})}
            for name, rows in [('development', self.development), ('holdout', self.holdout)]
        })
        atomic_json(self.root / 'manifest.json', {
            'model': 'test-student', 'case_ids': [x['record']['id'] for x in self.development],
            'benchmark_hash': digest(self.development),
            'checkpoints': {s: {'config': 'a' * 64, 'weights': str(i) * 64}
                            for i, s in enumerate(('initial', 'sft', 'sft_orpo'))},
        })
        for stage in ('initial', 'sft', 'sft_orpo'):
            generations = [{'raw': json.dumps(x['reference']), 'truncated': False} for x in self.development]
            self.write_predictions(stage, generations)

    def write_predictions(self, stage, predictions):
        atomic_json(self.root / f'A_{stage}_generation.json', predictions)
        atomic_jsonl(self.root / f'A_{stage}_strict-reference.jsonl', [
            {**pred, 'scenario_id': item['record']['id'], 'task': item['record']['task'], 'task_score': 0}
            for item, pred in zip(self.development, predictions)
        ])

    def test_fully_local_and_ignores_strict_reference_scores(self):
        with patch('distillation_loop.semantic_eval.create_parent') as parent:
            report = run(self.root, self.split, self.output)
            parent.assert_not_called()
        self.assertEqual(report['max_additional_usd'], '0')
        self.assertEqual(report['unique_missing_judgments'], 0)
        self.assertEqual(report['measured_without_judging']['sft'],
                         {'schema_evidence_valid_rate': 1, 'incomplete_generation_rate': 0})
        comparison = json.loads((self.output / 'semantic_comparison.json').read_text())
        self.assertEqual(comparison['runs']['sft']['macro_task_score'], 1)
        self.assertEqual(comparison['diagnostic_decision']['outcome'], 'inconclusive')

    def test_rejects_bad_hash_and_incomplete_predictions(self):
        manifest = json.loads(self.split.read_text())
        bad = copy.deepcopy(manifest)
        bad['holdout']['sha256'] = 'wrong'
        atomic_json(self.split, bad)
        with self.assertRaisesRegex(ValueError, 'holdout benchmark hash'):
            run(self.root, self.split, self.output)
        atomic_json(self.split, manifest)
        self.write_predictions('sft', [])
        with self.assertRaisesRegex(ValueError, 'Incomplete Pass A'):
            run(self.root, self.split, self.output)

    def test_missing_judgment_requires_matching_approval_and_sufficient_cap(self):
        from distillation_loop.review import grade_reviewed

        predictions = [{'raw': json.dumps(item['reference']), 'truncated': False}
                       for item in self.development]
        for i, item in enumerate(self.development):
            if isinstance(item['reference'].get('answer'), str):
                response = {**item['reference'], 'answer': item['reference']['answer'] + ' Additional text.'}
                candidate = {'raw': json.dumps(response), 'truncated': False}
                if grade_reviewed(item, candidate, self.schemas, {})[0] is None:
                    predictions[i] = candidate
                    break
        else:
            self.fail('Fixture needs one valid prediction requiring a judge')
        self.write_predictions('sft', predictions)
        with patch('distillation_loop.semantic_eval.create_parent') as parent:
            report = run(self.root, self.split, self.output)
            self.assertEqual(report['unique_missing_judgments'], 1)
            self.assertFalse((self.output / 'semantic_comparison.json').exists())
            with self.assertRaisesRegex(ValueError, 'Approval does not match'):
                run(self.root, self.split, self.output, approval='old', max_usd='100')
            with self.assertRaisesRegex(ValueError, 'cap must cover'):
                run(self.root, self.split, self.output, approval=report['approval_hash'], max_usd='0')
            parent.assert_not_called()
