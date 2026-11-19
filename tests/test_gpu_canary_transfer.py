"""Authorization and immutable-transfer checks; never loads torch/model weights."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gpu_canary_runner', ROOT / 'scripts/gpu_canary_runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CanaryTransferTests(unittest.TestCase):
    def args(self, **overrides):
        values = dict(authorize_training=True, provider='confirmed provider', expected_gpu='approved GPU',
                      billing_mode='paid', max_gpu_usd=1., gpu_hourly_rate_usd=.5,
                      max_runtime_seconds=1200.)
        return SimpleNamespace(**{**values, **overrides})

    def test_authorization_rejects_missing_consent_hardware_change_and_budget_increase(self):
        for override in [dict(authorize_training=False), dict(provider=''), dict(expected_gpu='different GPU'),
                         dict(max_gpu_usd=1.01), dict(max_gpu_usd=float('nan')),
                         dict(gpu_hourly_rate_usd=0), dict(gpu_hourly_rate_usd=float('inf')),
                         dict(max_runtime_seconds=1201), dict(billing_mode=None)]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                runner.authorization(self.args(**override), {'gpu': 'approved GPU'})

    def test_free_allocation_needs_zero_actual_rate(self):
        runner.authorization(self.args(billing_mode='free', gpu_hourly_rate_usd=0), {'gpu': 'approved GPU'})
        with self.assertRaises(ValueError):
            runner.authorization(self.args(billing_mode='free', gpu_hourly_rate_usd=.5), {'gpu': 'approved GPU'})

    def test_wrong_archive_stops_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive = base / 'wrong.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('job.json', '{}')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                runner.extract(archive, base / 'extracted')
            self.assertFalse((base / 'extracted').exists())

    def test_export_preserves_failure_evidence_without_success_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / 'package'
            root.mkdir()
            runner.save(base / 'training-observations.json', {'status': 'failed', 'failure': 'CUDA OOM'})
            (base / 'training.log').write_text('CUDA OOM\n')
            receipt = runner.export(root)
            self.assertEqual(receipt['sha256'], runner.sha(receipt['archive']))
            with zipfile.ZipFile(receipt['archive']) as z:
                self.assertIn('diagnostics/training.log', z.namelist())
                self.assertNotIn('result/result.json', z.namelist())
            with self.assertRaisesRegex(ValueError, 'already exists'):
                runner.export(root)

    def test_notebook_embeds_current_runner_with_unexecuted_cells(self):
        import ast
        n = json.loads((ROOT / 'notebooks/qwen3_8b_continued_gpu_canary.ipynb').read_text())
        embedded = None
        for c in n['cells']:
            if c['cell_type'] != 'code':
                continue
            self.assertIsNone(c['execution_count'])
            self.assertEqual(c['outputs'], [])
            tree = ast.parse(c['source'])
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'RUNNER' and node.func.attr == 'write_text':
                    embedded = ast.literal_eval(node.args[0])
        self.assertEqual(embedded, (ROOT / 'scripts/gpu_canary_runner.py').read_text())

    def test_observation_wrapper_runs_existing_worker_paths_without_gpu(self):
        # Exercise authorization, observation hooks, receipt writing and cleanup
        # with simulated worker outputs, not a model or an alternate trainer.
        import sys
        from autonomous_learning import gpu_worker
        from distillation_loop import student, adapter_identity

        class Runtime:
            model = SimpleNamespace(is_loaded_in_4bit=True)
            def identity(self): return {'model_revision': runner.REVISION}
            def prepare(self): pass
            def restore(self, path): pass
            def infer(self, payload): return {'raw': '{"smoke":true}'}

        class Trainer:
            state = SimpleNamespace(global_step=2, log_history=[{'loss': 1.0}, {'loss': .9}])
            def train(self): return SimpleNamespace(training_loss=.95)

        cuda = SimpleNamespace(reset_peak_memory_stats=lambda n: None,
                               max_memory_allocated=lambda n: 123, max_memory_reserved=lambda n: 456)
        fake_torch = SimpleNamespace(cuda=cuda)
        fake_trl = SimpleNamespace(SFTTrainer=Trainer)

        def core(root, **kwargs):
            runtime = student.StudentRuntime()
            runtime.prepare()
            runtime.restore(root / 'initial_adapter')
            fake_trl.SFTTrainer().train()
            output = root / 'result'
            (output / 'adapter').mkdir(parents=True)
            for name in runner.ADAPTER:
                (output / 'adapter' / name).write_bytes(b'simulated file')
            runtime.restore(output / 'adapter')
            runtime.infer({'task': 'simulated observation fixture'})
            return {'adapter_files': runner.ADAPTER, 'adapter_reload_verified': True,
                    'minimal_inference_verified': True}

        for mode, rate in [('free', 0), ('paid', .5)]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp); root = base / 'package'; root.mkdir()
                runner.save(base / 'cloud-preflight.json', {'status': 'passed', 'job_id': runner.JOB})
                job = {'output_directory': 'result'}
                with mock.patch.dict(sys.modules, {'torch': fake_torch, 'trl': fake_trl}), \
                     mock.patch.object(runner, 'verify', return_value=job), \
                     mock.patch.object(runner, 'hardware', return_value={'gpu': 'approved GPU'}), \
                     mock.patch.object(student, 'StudentRuntime', Runtime), \
                     mock.patch.object(adapter_identity, 'adapter_files', return_value=runner.ADAPTER), \
                     mock.patch.object(gpu_worker, '_run', side_effect=core) as free_worker, \
                     mock.patch.object(gpu_worker, 'run', side_effect=core) as paid_worker:
                    result = runner.train(root, self.args(billing_mode=mode, gpu_hourly_rate_usd=rate))
                    self.assertEqual(free_worker.call_count, int(mode == 'free'))
                    self.assertEqual(paid_worker.call_count, int(mode == 'paid'))
                    if mode == 'paid':
                        self.assertEqual(paid_worker.call_args.kwargs['gpu_hourly_rate_usd'], .5)
                saved = json.loads((root / 'result/result.json').read_text())
                observed = saved['canary_observations']
                self.assertEqual(observed['optimizer_steps'], 2)
                self.assertEqual(observed['peak_allocated_vram_bytes'], 123)
                self.assertEqual(observed['status'], 'completed')
                self.assertTrue(observed['inference']['prediction']['raw'])
                if mode == 'free': self.assertEqual(result['estimated_gpu_cost_usd'], 0)


if __name__ == '__main__':
    unittest.main()
