"""Notebook coordinator: model weights live only in fresh torchrun GPU workers."""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from self_instruct_pipeline.config import ROOT
from self_instruct_pipeline.persistence import atomic_json, append_jsonl, read_jsonl


def worker_command(job_path, processes=2):
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            f'--nproc_per_node={processes}', '--module', 'distillation_loop.gpu_worker', str(job_path)]


def merge_predictions(rows, expected):
    by_index = {r['index']: r['prediction'] for r in rows}
    if len(rows) != expected or len(by_index) != expected or set(by_index) != set(range(expected)):
        raise ValueError('GPU workers returned missing/duplicate prediction indices')
    return [by_index[i] for i in range(expected)]


class DualGPURuntime:
    """Two-GPU DDP training and sharded batched inference; no DataParallel."""
    model = None  # The notebook must not retain an extra 8B model on GPU 0.

    def __init__(self, tokenizer, config, model_name='Qwen/Qwen3-8B', batch_size=2, timeout_seconds=7200):
        if type(batch_size) is not int or batch_size < 1 or timeout_seconds <= 0:
            raise ValueError('Invalid worker batch size or timeout')
        from transformers import AutoConfig
        self.tokenizer, self.config, self.model_name = tokenizer, config, model_name
        self.model_config = AutoConfig.from_pretrained(model_name, token=False)
        self.batch_size, self.timeout_seconds = batch_size, timeout_seconds
        self.checkpoint = None
        self.prepared = False

    def identity(self):
        return {'model_name': self.model_config._name_or_path,
                'model_revision': getattr(self.model_config, '_commit_hash', None),
                'tokenizer_name': self.tokenizer.name_or_path,
                'tokenizer_revision': self.tokenizer.init_kwargs.get('_commit_hash'),
                'tokenizer_class': type(self.tokenizer).__name__}

    def prepare(self):
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise ValueError('Dual-GPU mode requires two visible GPUs. Restart and set CUDA_VISIBLE_DEVICES=0,1 before importing torch.')
        if any(torch.cuda.memory_allocated(i) for i in range(2)):
            raise ValueError('Restart the kernel and use the dual-GPU model cell: notebook model weights must not occupy the worker GPUs.')
        if self.config['training']['gradient_accumulation_steps'] % 2:
            raise ValueError('Dual-GPU training requires an even gradient_accumulation_steps')
        self.prepared = True

    def restore(self, checkpoint):
        path = Path(checkpoint).resolve()
        if not (path/'adapter_config.json').exists():
            raise ValueError('Missing adapter checkpoint')
        self.checkpoint = str(path)

    def save_checkpoint(self, checkpoint):
        checkpoint = Path(checkpoint).resolve()
        if self.checkpoint:
            if Path(self.checkpoint) != checkpoint:
                shutil.copytree(self.checkpoint, checkpoint, dirs_exist_ok=True)
        else:
            self._run('initialize', checkpoint=str(checkpoint), processes=1)
            self.checkpoint = str(checkpoint)
        return str(checkpoint)

    def _run(self, operation, processes=2, **payload):
        self.prepare()
        jobs = Path(self.config['output_dir']).resolve()/'gpu_jobs'
        jobs.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=jobs) as folder:
            directory = Path(folder)
            job = {'operation': operation, 'config': self.config, 'model_name': self.model_name,
                   'adapter': self.checkpoint, 'batch_size': self.batch_size, **payload}
            atomic_json(directory/'job.json', job)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='0,1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
            env.setdefault('NCCL_P2P_DISABLE', '1')
            env.setdefault('NCCL_IB_DISABLE', '1')
            for key in ('OPENAI_API_KEY', 'GEMINI_API_KEY', 'SUPABASE_SECRET_KEY', 'KAGGLE_API_TOKEN', 'KAGGLE_KEY'):
                env.pop(key, None)
            started = time.monotonic()
            print(f'Starting {operation} on {processes} GPU worker(s); loading cached model weights.', flush=True)
            process = subprocess.Popen(worker_command(directory/'job.json', processes), cwd=ROOT,
                                       env=env, start_new_session=True)
            try:
                code = process.wait(timeout=self.timeout_seconds)
                if code:
                    raise RuntimeError(f'GPU {operation} failed (exit {code}); inspect worker traceback above. No round checkpoint committed.')
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
            duration = time.monotonic() - started
            append_jsonl(jobs.parent/'hardware_metrics.jsonl', {
                'operation': operation, 'seconds': duration, 'gpu_workers': processes,
                'inference_batch_size_per_gpu': self.batch_size,
                'examples': len(payload.get('payloads', payload.get('examples', [])))})
            print(f'{operation} finished in {duration:.1f}s', flush=True)
            if operation == 'infer':
                rows = [row for rank in range(processes)
                        for row in read_jsonl(directory/f'predictions_{rank}.jsonl')]
                return merge_predictions(rows, len(payload['payloads']))

    def infer(self, payload):
        return self.infer_many([payload])[0]

    def infer_many(self, payloads):
        if not payloads:
            return []
        return self._run('infer', payloads=payloads)

    def train(self, pairs, schemas, output_dir):
        from .student import tokenize_teacher_pair
        from self_instruct_pipeline.prompt_builder import target_payload
        examples = [tokenize_teacher_pair(self.tokenizer, target_payload(record, schemas), answer,
                    self.config['training']['max_sequence_length']) for record, answer in pairs]
        self._run('train', examples=examples, output_dir=str(Path(output_dir).resolve()))
        checkpoint = Path(output_dir).resolve()/'adapter'
        self.restore(checkpoint)
        return str(checkpoint)
