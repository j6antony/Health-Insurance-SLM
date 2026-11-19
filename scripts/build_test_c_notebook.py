"""Create a runnable Test C notebook bound to the frozen transfer receipt."""
import hashlib
import json
from pathlib import Path
import textwrap
import argparse


def build(receipt_path, output):
    receipt = json.loads(Path(receipt_path).read_text())
    cells = []
    def cell(kind, text):
        value = {'cell_type': kind, 'metadata': {}, 'source': textwrap.dedent(text).strip() + '\n'}
        value['id'] = hashlib.sha256((str(len(cells)) + value['source']).encode()).hexdigest()[:12]
        if kind == 'code': value.update(execution_count=None, outputs=[])
        cells.append(value)
    cell('markdown', '''
        # Test C — frozen baseline vs Test B candidate
        Inference only: 28 cases × two immutable rank-32 RSLoRA adapters. No training or promotion.
        Same-generator synthetic development, not independent probes or a release holdout.
        User approved one Colab L4, at most US$1 worker credit valuation and 60 minutes.
        Confirm the visible allocation/rate before execution. Download and verify results before deleting the runtime.
        Original adapter history is incomplete: leakage checks cover all available seven-task training/replay exports.
    ''')
    cell('code', '''
        import os, sys, json, subprocess, shutil, hashlib, zipfile, platform
        from pathlib import Path
        assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        BASE = Path('/content/test-c')
        BASE.mkdir(exist_ok=True)
        ROOT = BASE / 'package'
        VENV = BASE / 'venv'
        PY = VENV / 'bin/python'
        os.environ['HF_HOME'] = str(BASE / 'hf-cache')
        def execute(command, name):
            log = BASE / name
            if log.exists(): raise RuntimeError(f'Preserve existing log/attempt: {log}')
            with log.open('w') as f:
                p = subprocess.Popen([str(x) for x in command], cwd=ROOT if ROOT.exists() else BASE,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                try:
                    for line in p.stdout:
                        print(line, end=''); f.write(line); f.flush()
                    code = p.wait()
                except BaseException:
                    p.terminate()
                    try: p.wait(timeout=10)
                    except subprocess.TimeoutExpired: p.kill(); p.wait()
                    raise
            if code: raise RuntimeError(f'Failed ({code}): inspect {log}; never regenerate completed predictions.')
        print(subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                     '--format=csv,noheader'], text=True))
    ''')
    cell('markdown', '## Upload the exact package; its checksum is verified before extraction.')
    cell('code', f'''
        from google.colab import files
        previous = Path.cwd(); os.chdir(BASE)
        try: uploaded = files.upload()
        finally: os.chdir(previous)
        if len(uploaded) != 1: raise RuntimeError('Upload exactly one Test C ZIP.')
        archive = BASE / next(iter(uploaded)); del uploaded
        h = hashlib.sha256()
        with archive.open('rb') as f:
            for b in iter(lambda: f.read(1048576), b''): h.update(b)
        assert h.hexdigest() == {receipt['sha256']!r}, 'Package SHA mismatch'
        assert not ROOT.exists(), 'Preserve existing extraction'
        with zipfile.ZipFile(archive) as z:
            assert len(z.namelist()) == len(set(z.namelist()))
            for i in z.infolist():
                assert (ROOT / i.filename).resolve().is_relative_to(ROOT.resolve())
                assert (i.external_attr >> 16) & 0o170000 != 0o120000
            z.extractall(ROOT)
        execute([sys.executable, '-c', 'from autonomous_learning.paired_evaluation import verify_package; from pathlib import Path; j,b=verify_package(Path(".")); print(j["job_id"], len(b["cases"]))'], 'package-verification.log')
    ''')
    cell('markdown', '## Isolated Python/CUDA environment — no full model loading')
    cell('code', '''
        assert shutil.disk_usage(BASE).free >= 40 * 1024**3
        assert not VENV.exists(), 'Preserve earlier environment attempt'
        compatible = None
        for candidate in dict.fromkeys(x for x in [sys.executable, shutil.which('python3.12'), shutil.which('python3.11')] if x):
            version = json.loads(subprocess.check_output([candidate, '-c', 'import sys,json; print(json.dumps(list(sys.version_info[:2])))'], text=True))
            if (3,10) <= tuple(version) <= (3,12): compatible = candidate; break
        if compatible:
            execute([compatible, '-m', 'venv', '--without-pip', VENV], 'venv.log')
            execute([sys.executable, '-m', 'pip', '--python', PY, 'install', 'pip'], 'pip-bootstrap.log')
        else:
            bootstrap = BASE / 'uv-bootstrap'
            execute([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', bootstrap, 'uv==0.9.11'], 'uv-bootstrap.log')
            uv = bootstrap / 'bin/uv'
            os.environ['UV_PYTHON_INSTALL_DIR'] = str(BASE / 'managed-python')
            os.environ['UV_CACHE_DIR'] = str(BASE / 'uv-cache')
            execute([uv, 'python', 'install', '3.12.11'], 'python-install.log')
            execute([uv, 'venv', '--python', '3.12.11', '--seed', VENV], 'venv.log')
        execute([PY, '-m', 'pip', 'install', '--no-cache-dir', 'torch==2.8.0', 'torchvision==0.23.0', 'torchaudio==2.8.0', '--index-url', 'https://download.pytorch.org/whl/cu126'], 'torch-install.log')
        execute([PY, '-m', 'pip', 'install', '--no-cache-dir', '-c', ROOT / 'constraints.txt', '-r', ROOT / 'requirements-training.txt', 'tokenizers==0.22.2', 'huggingface-hub==0.36.0', 'safetensors==0.6.2'], 'dependencies-install.log')
        execute([PY, '-m', 'pip', 'check'], 'pip-check.log')
        execute([PY, '-m', 'pip', 'freeze'], 'pip-freeze.log')
        execute([PY, '-c', 'import sys,torch; print(sys.version); print(torch.__version__); print(torch.cuda.get_device_name(0)); assert torch.cuda.get_device_name(0)=="NVIDIA L4"'], 'gpu-check.log')
    ''')
    cell('markdown', '## Tokenizer/contract preflight — exact frozen references, no truncation')
    cell('code', '''
        execute([PY, '-m', 'autonomous_learning.evaluation_worker', ROOT, '--validate-only'], 'preflight.log')
    ''')
    cell('markdown', '''
        ## Authorized Test C paired inference
        Run only after the separate Test C authorization and preflight pass.
        The rate below values existing credits using the disclosed CA$13.99/100 CU retail basis,
        1.94 CU/hour displayed for the L4, and latest published CAD→USD 0.6980 (2026-11-18).
        Confirm the current Colab resource panel still reports that allocation/rate before running.
        At most 60 worker minutes and US$1. No API generation, training, benchmark regeneration or promotion.
        Raw predictions are saved after each case and are never repaired or regenerated on resume.
    ''')
    cell('code', '''
        rate = .1399 * 1.94 * .6980
        approval = {'operation': 'Test C inference only', 'provider': 'Google Colab Pro', 'gpu': 'NVIDIA L4',
                    'max_gpu_usd': 1, 'max_seconds': 3600, 'valuation_usd_per_hour': rate,
                    'basis': 'Disclosed retail valuation of existing credits; not invoice-confirmed new spending',
                    'authorization': 'User approved Test C within one-L4, USD1, 60-minute limits.'}
        (BASE / 'test-c-authorization.json').write_text(json.dumps(approval, indent=2))
        execute([PY, '-m', 'autonomous_learning.evaluation_worker', ROOT, '--authorize-inference',
                 '--gpu-hourly-rate-usd', str(rate), '--max-gpu-usd', '1', '--max-seconds', '3600'], 'paired-inference.log')
        print((BASE / 'results/metrics.json').read_text())
    ''')
    cell('markdown', '## Return complete results, verify on the SSD, then disconnect/delete the GPU runtime')
    cell('code', '''
        from google.colab import files
        result_files = {str(p.relative_to(BASE)): p for p in (BASE / 'results').rglob('*') if p.is_file()}
        result_files.update({p.name: p for p in BASE.iterdir() if p.is_file() and p.suffix in ('.json','.log')})
        result_files.update({'evaluation-job.json': ROOT / 'evaluation-job.json', 'benchmark.json': ROOT / 'benchmark.json', 'benchmark-manifest.json': ROOT / 'benchmark-manifest.json'})
        checks = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name,p in result_files.items()}
        archive = BASE / 'test-c-return.zip'
        assert not archive.exists(), 'Preserve previous return archive'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('return-manifest.json', json.dumps(checks, indent=2))
            for name,p in sorted(result_files.items()): z.write(p,name)
        print('RETURN_SHA256', hashlib.sha256(archive.read_bytes()).hexdigest())
        files.download(str(archive))
        print('Verify the download before disconnecting and deleting the runtime.')
    ''')
    notebook = {'nbformat':4, 'nbformat_minor':5, 'metadata': {'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                'language_info':{'name':'python'}, 'accelerator':'GPU', 'colab':{'name':Path(output).name}}, 'cells':cells}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(notebook,indent=2)+'\n')
    return str(output)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt');p.add_argument('output');a=p.parse_args()
    print(build(a.receipt,a.output))
