import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from autonomous_learning.scenarios import make_scenario
from autonomous_learning.rendering import render
from autonomous_learning.tasks import project, TASKS
from autonomous_learning.storage import Storage, digest, atomic_json
from autonomous_learning.evaluation import import_benchmark, DEVELOPMENT_PROTOCOL
from autonomous_learning.paired_evaluation import development_leakage, freeze, diagnose


class DevelopmentEvaluationTests(unittest.TestCase):
    def rows(self, seed=612):
        s = make_scenario(seed, 0, 'hospitalization', True, 'valid', 0)
        return project(s, render(s))

    def benchmark(self, rows, **overrides):
        return {**{'version': 2, 'tier': 'development', 'process': 'in_generator',
                   'independently_reviewed': False, 'development_protocol': DEVELOPMENT_PROTOCOL,
                   'cases': rows}, **overrides}

    def test_shared_template_exception_cannot_escape_development(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = Storage(tmp); storage.initialize()
            rows = self.rows()[:1]
            scenario = make_scenario(999, 0, 'pharmacy', False, 'incomplete', 4)
            training = project(scenario, render(scenario))
            self.assertEqual(import_benchmark(storage, self.benchmark(rows), training)['case_count'], 1)
            for changes in [{'tier': 'probes'}, {'tier': 'release_holdout'},
                            {'process': 'independent'}, {'independently_reviewed': True}]:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    import_benchmark(storage, self.benchmark(rows, **changes))

    def test_original_strict_template_rule_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = Storage(tmp); storage.initialize()
            rows = self.rows(); training = self.rows(999)
            strict = self.benchmark(rows, version=1); strict.pop('development_protocol')
            with self.assertRaisesRegex(ValueError, 'template-family'):
                import_benchmark(storage, strict, training)

    def test_family_and_normalized_near_duplicates_cannot_be_relabeled_away(self):
        training = self.rows()
        with self.assertRaisesRegex(ValueError, 'family'):
            development_leakage(training, training)
        altered = copy.deepcopy(training[0])
        altered.update(id='different-id', family_id='different-family', template_family='different-template')
        with self.assertRaisesRegex(ValueError, 'near-duplicate'):
            development_leakage(training, [altered])

    def test_references_are_reconstructed_not_merely_offset_checked(self):
        rows = self.rows(); rows[0]['target']['requires_review'] = True
        with tempfile.TemporaryDirectory() as tmp:
            storage = Storage(tmp); storage.initialize()
            with self.assertRaisesRegex(ValueError, 'Target inconsistent'):
                import_benchmark(storage, self.benchmark(rows))

    def test_freeze_resume_never_regenerates_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = Storage(tmp); storage.initialize()
            training = Path(tmp) / 'training.jsonl'
            training.write_text(''.join(json.dumps(r) + '\n' for r in self.rows()))
            benchmark, manifest = freeze(storage, [training])
            self.assertEqual(len(benchmark['cases']), 28)
            self.assertEqual(set(manifest['per_task']), set(TASKS))
            self.assertTrue(all(n == 4 for n in manifest['per_task'].values()))
            resumed, same = freeze(storage, [training])
            self.assertEqual(benchmark, resumed); self.assertEqual(manifest, same)
            path = Path(tmp) / 'experiments/test-c-baseline-comparison/benchmark.json'
            altered = copy.deepcopy(benchmark); altered['cases'][0]['target']['requires_review'] = True
            atomic_json(path, altered)
            with self.assertRaisesRegex(ValueError, 'changed'):
                freeze(storage, [training])

    def test_diagnostics_keep_unknown_free_text_semantics_unavailable(self):
        from autonomous_learning.evaluation import score_case
        case = self.rows()[0]
        prediction = {'raw': json.dumps(case['target']), 'truncated': False}
        score = score_case(case, prediction)
        diagnostic = diagnose(case, prediction, score)
        self.assertEqual(score['score'], 1)
        self.assertEqual(diagnostic['categories'], [])
        self.assertIsNone(diagnostic['free_text_assertion_verified'])
        malformed = {'raw': 'not json', 'truncated': False}
        diagnostic = diagnose(case, malformed, score_case(case, malformed))
        self.assertIn('invalid_output_json', diagnostic['categories'])
        self.assertIsNone(diagnostic['structured_unsupported'])

    def test_paired_scoring_detects_one_review_regression_without_claiming_semantics(self):
        from autonomous_learning.paired_evaluation import analyze
        rows = []
        for ti in range(7):
            for ci, category in enumerate(('hospitalization', 'outpatient', 'pharmacy', 'diagnostic')):
                s = make_scenario(7462, ti * 4 + ci, category, True, 'valid', ci)
                rows.append(project(s, render(s))[ti])
        benchmark = self.benchmark(rows)
        predictions = {c['id']: {'raw': json.dumps(c['target']), 'truncated': False} for c in rows}
        baseline = {'benchmark_id': digest(benchmark), 'adapter_sha256': digest({'fixture':'hash'}),
                    'contract_version': 'document-verification-v1', 'model_name': 'Qwen/Qwen3-8B',
                    'inference_config': {'fixture': 'identical'}, 'predictions': predictions}
        candidate = copy.deepcopy(baseline)
        changed = copy.deepcopy(rows[0]['target']); changed['requires_review'] = not changed['requires_review']
        candidate['predictions'][rows[0]['id']]['raw'] = json.dumps(changed)
        with tempfile.TemporaryDirectory() as tmp, mock.patch('autonomous_learning.evaluation.adapter_files', return_value={'fixture':'hash'}):
            report = analyze(benchmark, {'generations':baseline,'checkpoint':'fixture'},
                             {'generations':candidate,'checkpoint':'fixture'}, tmp)
            self.assertEqual(report['metrics']['baseline']['macro_task_score'], 1)
            self.assertAlmostEqual(report['metrics']['candidate']['macro_task_score'], 27/28)
            self.assertEqual(report['comparison']['paired']['losses'], 1)
            self.assertEqual(report['comparison']['paired']['ties'], 27)
            self.assertIsNone(report['metrics']['candidate']['unsupported_assertion_rate'])
            feedback=json.loads((Path(tmp)/'curriculum-feedback.json').read_text())
            self.assertEqual(feedback['planner_input']['measured_priority'][0]['task'], rows[0]['task'])


if __name__ == '__main__': unittest.main()
