import ast
import hashlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from self_instruct_pipeline.config import ROOT
import test_distillation_loop as fixtures
from test_distillation_loop import FakeRuntime, SCHEMAS, report


class ReviewRoundOneTests(unittest.TestCase):
    def fixture(self):
        fixture = fixtures.ControllerTests()
        fixture.setUp()
        self.addCleanup(fixture.tmp.cleanup)
        fixture.config['max_rounds'] = 1
        return fixture

    def test_sft_receipt_resume_restores_candidate_before_inference(self):
        from distillation_loop import controller

        fixture = self.fixture()
        class StateRuntime(FakeRuntime):
            def restore(self, path):
                super().restore(path)
                self.loaded = (Path(path) / 'adapter_model.safetensors').read_bytes()

            def train(self, pairs, schemas, output_dir):
                path = super().train(pairs, schemas, output_dir)
                (Path(path) / 'adapter_model.safetensors').write_bytes(b'candidate')
                self.loaded = b'candidate'
                return path

            def infer(self, payload):
                if payload.get('probe'):
                    loaded.append(self.loaded)
                return super().infer(payload)

        loaded = []
        class Evaluator:
            def evaluate(self, development, directory):
                runtime.infer({'probe': True})
                return report(.5 if directory.name == 'baseline' else .7)

        runtime = StateRuntime()
        save = controller.atomic_json
        def interrupt(path, data):
            if path.name == 'training_complete.json':
                raise RuntimeError('interrupted after SFT receipt')
            save(path, data)

        with patch('distillation_loop.controller.atomic_json', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'interrupted after SFT receipt'):
                fixture.run_loop(runtime, Evaluator())
        self.assertTrue((Path(fixture.tmp.name) / 'round_01/sft_complete.json').is_file())
        # A new process starts with the initial adapter.
        runtime = StateRuntime()
        fixture.run_loop(runtime, Evaluator())
        self.assertEqual(runtime.trains, [])
        self.assertEqual(loaded, [b'test fixture weights', b'candidate'])

    def test_legacy_release_cannot_promote_aggregate_only_reports(self):
        from distillation_loop.quality import promotion_gate
        thresholds = dict(macro_task_score=.9, minimum_task_score=.8, schema_evidence_valid_rate=.98)
        result = promotion_gate(report(.5), report(1), {}, thresholds, SCHEMAS)
        self.assertFalse(result['accepted'])
        self.assertIn('evidence', result['reason'])
        baseline, selected = map(fixtures.with_safety_evidence, (report(.5), report(1)))
        for field in ('benchmark_hash', 'scoring', 'evaluation_protocol'):
            result = promotion_gate({**baseline, field: ''}, {**selected, field: ''}, {}, thresholds, SCHEMAS)
            self.assertFalse(result['accepted'])
            self.assertIn(field, result['reason'])
        renamed = dict(selected['case_scores'])
        renamed['other'] = renamed.pop(next(iter(renamed)))
        result = promotion_gate(baseline, {**selected, 'case_scores': renamed}, {}, thresholds, SCHEMAS)
        self.assertFalse(result['accepted'])
        self.assertIn('case IDs', result['reason'])

    def test_cluster_bootstrap_uses_sampled_family_sizes_and_macro_task_weights(self):
        from distillation_loop.paired import paired_summary
        values = [1., 0., 0., -1.]
        report_data = paired_summary(values, ['small', 'large', 'large', 'other'],
                                     strata=['first', 'first', 'first', 'second'])
        self.assertAlmostEqual(report_data['delta'], -1 / 3)
        self.assertEqual(report_data['source_families'], 3)
        self.assertEqual(report_data['clustered_bootstrap_95pct_ci'], [-.5, 0])
        self.assertEqual(report_data['bootstrap_seed'], 42)

    def test_verified_colab_replaces_stale_extracted_code(self):
        notebook = json.loads((ROOT / 'semantic_diagnostic_colab.ipynb').read_text())
        main = ast.parse(''.join(notebook['cells'][1]['source']))
        helper = next((node for node in main.body if isinstance(node, ast.FunctionDef)
                       and node.name == 'verified_project'), None)
        self.assertIsNotNone(helper, 'Expected notebook bundle verification helper')
        with TemporaryDirectory() as folder:
            root = Path(folder)
            bundle = root / 'bundle.zip'
            project = root / 'Health-Insurance-SLM-Colab'
            project.mkdir()
            target = project / 'distillation_loop/semantic_eval.py'
            target.parent.mkdir()
            target.write_text('old')
            (project / 'untracked.py').write_text('stale')
            with ZipFile(bundle, 'w') as archive:
                archive.writestr(f'{project.name}/distillation_loop/semantic_eval.py', 'verified')
            from tempfile import TemporaryDirectory as Temp, mkdtemp
            namespace = {'Path': Path, 'hashlib': hashlib, 'zipfile': __import__('zipfile'),
                         'TemporaryDirectory': Temp, 'mkdtemp': mkdtemp, 'BytesIO': BytesIO}
            exec(compile(ast.Module(body=[helper], type_ignores=[]), '<notebook>', 'exec'), namespace)
            verified = namespace['verified_project']
            checksum = hashlib.sha256(bundle.read_bytes()).hexdigest()
            self.assertEqual(verified(project, bundle, checksum), project)
            self.assertEqual(target.read_text(), 'verified')
            self.assertFalse((project / 'untracked.py').exists())
            bundle.write_bytes(b'bad zip')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verified(project, bundle, checksum)

    def test_experimental_github_backup_is_blocked_without_redirect(self):
        from distillation_loop.controller import backup_adapter
        with TemporaryDirectory() as folder, patch('distillation_loop.drive_backup.save_drive_adapter') as drive, \
                patch('distillation_loop.github_backup.upload_latest_adapter') as github:
            root = Path(folder)
            backup_adapter('github', root / 'adapter', root, 1, experimental=True)
            status = json.loads((root / 'round_01/backup_status.json').read_text())
            self.assertEqual((status['destination'], status['status']), ('github', 'blocked'))
            drive.assert_not_called()
            github.assert_not_called()


    def test_generation_cache_rejects_missing_identity_and_mutated_weights(self):
        from distillation_loop.evaluation import LightevalEvaluator
        from self_instruct_pipeline.persistence import atomic_json
        from test_quality_hardening import reasoning_item

        item = reasoning_item()
        class Runtime(FakeRuntime):
            def infer(self, payload):
                self.inferences.append(payload)
                return {'raw': json.dumps(item['reference']), 'truncated': False}

        with TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = Runtime()
            adapter = runtime.save_checkpoint(root / 'adapter')
            from distillation_loop.adapter_identity import file_sha256
            evaluator = LightevalEvaluator(runtime, {'policy_coverage_qa': SCHEMAS['policy_coverage_qa']}, 1)
            output = root / 'benchmark'
            with patch('distillation_loop.adapter_identity.file_sha256', wraps=file_sha256) as hash_file:
                runtime.restore(adapter)
                evaluator.evaluate([item], output)
                evaluator.evaluate([item], output)
                self.assertEqual(hash_file.call_count, 2)
            self.assertEqual(len(runtime.inferences), 1)
            generation = output / 'generation.json'
            saved = json.loads(generation.read_text())
            legacy = {key: value for key, value in saved.items() if key != 'adapter_sha256'}
            atomic_json(generation, legacy)
            with self.assertRaisesRegex(ValueError, 'lack verified adapter-weight identity'):
                evaluator.evaluate([item], output)
            atomic_json(generation, saved)
            (Path(adapter) / 'adapter_model.safetensors').write_bytes(b'changed adapter')
            with self.assertRaisesRegex(ValueError, 'does not match the runtime restore'):
                evaluator.evaluate([item], output)
            runtime.restore(adapter)
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                evaluator.evaluate([item], output)
            self.assertEqual(len(runtime.inferences), 1)

    def test_drive_archive_labels_local_path_and_rejects_unmounted_destination(self):
        from distillation_loop.drive_backup import save_drive_adapter
        with TemporaryDirectory() as folder:
            root = Path(folder)
            adapter = FakeRuntime().save_checkpoint(root / 'adapter')
            with self.assertRaisesRegex(ValueError, 'mounted MyDrive'):
                save_drive_adapter(adapter, root, 1, require_mounted=True)
            local = save_drive_adapter(adapter, root, 1)
            self.assertEqual(local['status'], 'local_archive_verified')
            self.assertEqual(local['destination'], 'local_filesystem')
