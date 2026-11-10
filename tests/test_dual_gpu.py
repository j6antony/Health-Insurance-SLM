import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from distillation_loop.dual_gpu import worker_command, merge_predictions
from distillation_loop.student import decode_prediction, StudentRuntime


class DualGPUTests(unittest.TestCase):
    def test_launch_is_process_per_gpu_not_data_parallel(self):
        command = worker_command('/tmp/job.json')
        self.assertIn('--nproc_per_node=2', command)
        self.assertIn('distillation_loop.gpu_worker', command)
        self.assertEqual(command[-1], '/tmp/job.json')

    def test_results_reordered_and_missing_duplicates_refused(self):
        rows = [{'index': 1, 'prediction': {'raw':'b'}}, {'index': 0, 'prediction': {'raw':'a'}}]
        self.assertEqual([r['raw'] for r in merge_predictions(rows, 2)], ['a','b'])
        for bad in [rows[:1], [rows[0], rows[0]], rows + rows[:1]]:
            with self.assertRaises(ValueError): merge_predictions(bad, 2)

    def test_eos_before_batch_padding_is_not_truncated(self):
        tokenizer = SimpleNamespace(eos_token_id=9, decode=lambda ids, **k: str(ids))
        row = decode_prediction(tokenizer, [2,9,0,0], 4)
        self.assertFalse(row['truncated'])
        self.assertEqual(row['generated_tokens'], 2)
        self.assertTrue(decode_prediction(tokenizer, [2,3,4,5], 4)['truncated'])
        self.assertFalse(decode_prediction(tokenizer, [2,3,4,9], 4)['truncated'])

    def runtime(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs): return messages[0]['content']
            def __call__(self, text, **kwargs): return {'input_ids': list(range(len(text)))}
        runtime = object.__new__(StudentRuntime)
        runtime.tokenizer = Tokenizer()
        runtime.config = {'inference': {'max_new_tokens': 8}, 'training': {'max_sequence_length': 100}}
        runtime.prepare = lambda: None
        return runtime

    def test_length_buckets_restore_original_order_and_reject_long_prompt(self):
        runtime = self.runtime(); batches=[]
        def generate(texts):
            batches.append(texts)
            return [{'raw': t, 'truncated': False} for t in texts]
        runtime._generate_batch = generate
        payloads = [{'x':'long value'}, {'x':'a'}, {'x':'b'}, {'x':'z'*100}]
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=RuntimeError))
        with patch.dict('sys.modules', {'torch': fake_torch}):
            rows = runtime.infer_many(payloads, 2)
        self.assertEqual([json.loads(r['raw']) for r in rows[:3]], payloads[:3])
        self.assertIn('Prompt exceeds', rows[3]['error'])
        self.assertEqual([len(batch) for batch in batches], [2,1])

    def test_oom_retries_smaller_batches_without_losing_rows(self):
        runtime = self.runtime(); sizes=[]
        class OOM(RuntimeError): pass
        def generate(texts):
            sizes.append(len(texts))
            if len(texts)>1: raise OOM('mock OOM')
            return [{'raw':texts[0], 'truncated':False}]
        runtime._generate_batch=generate
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM, empty_cache=lambda: None))
        with patch.dict('sys.modules', {'torch': fake_torch}):
            rows = runtime.infer_many([{'x':'a'}, {'x':'b'}])
        self.assertEqual(sizes, [2,1,1])
        self.assertEqual([json.loads(r['raw'])['x'] for r in rows], ['a','b'])

    def coordinator(self, folder):
        from distillation_loop.dual_gpu import DualGPURuntime
        runtime = object.__new__(DualGPURuntime)
        runtime.config = {'output_dir': folder}
        runtime.model_name = 'Qwen/Qwen3-8B'
        runtime.checkpoint = None
        runtime.batch_size = 2
        runtime.timeout_seconds = 60
        runtime.prepare = lambda: None
        return runtime

    def test_worker_jobs_strip_secrets_and_merge_rank_outputs(self):
        import tempfile
        from pathlib import Path
        from self_instruct_pipeline.persistence import atomic_jsonl
        process = SimpleNamespace(wait=lambda **k: 0, poll=lambda: 0)
        def launch(command, **kwargs):
            for key in ('OPENAI_API_KEY','GEMINI_API_KEY','SUPABASE_SECRET_KEY'):
                self.assertNotIn(key, kwargs['env'])
            job_path = Path(command[-1])
            job = json.loads(job_path.read_text())
            self.assertEqual(job['operation'], 'infer')
            for rank in range(2):
                atomic_jsonl(job_path.parent/f'predictions_{rank}.jsonl',
                    [{'index': i, 'prediction': {'raw': str(i)}} for i in range(rank, len(job['payloads']), 2)])
            return process
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {'OPENAI_API_KEY':'SECRET','SUPABASE_SECRET_KEY':'SECRET'}), patch('distillation_loop.dual_gpu.subprocess.Popen', side_effect=launch):
            rows = self.coordinator(folder).infer_many([{'x':i} for i in range(5)])
            self.assertEqual([r['raw'] for r in rows], ['0','1','2','3','4'])
            self.assertTrue((Path(folder)/'hardware_metrics.jsonl').exists())

    def test_timeout_terminates_whole_worker_group(self):
        import tempfile
        import subprocess
        from unittest.mock import Mock
        process = Mock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired('mock', 60), 0]
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as folder, patch('distillation_loop.dual_gpu.subprocess.Popen', return_value=process), patch('distillation_loop.dual_gpu.os.killpg') as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                self.coordinator(folder).infer_many([{'x':1}])
            kill.assert_called_once()
