"""Build the self-contained upload notebook; no GPU or network calls."""
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def build():
    helper = (ROOT / 'scripts/gpu_canary_runner.py').read_text()
    cells = []

    def md(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(text).strip() + '\n'})

    def code(text):
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                      'source': textwrap.dedent(text).strip() + '\n'})

    md('''
        # Test B — continued Qwen3-8B adapter, two optimizer steps

        Upload this notebook into Colab or Kaggle. Use a Linux x86-64 Python 3.10–3.12 runtime.
        Do not allocate paid hardware until you approve its provider, GPU, actual hourly rate and budget.
        Select one 24 GB-class GPU (at least 22 GiB reported total, 20 GiB free), compute capability 7.x–9.x.
        Newer compute architectures need a separately reviewed CUDA stack. Two 16 GB GPUs are not a substitute.
        On Kaggle enable Internet and expose only GPU 0. If only a T4/P100 16 GB allocation is available, stop.

        The original 350.7 MB ZIP is uploaded separately; the cloud cannot access your Mac SSD.
        No fresh adapter or new dataset is created. Initial rank-32 RSLoRA is preserved.
        The saved legacy adapter has `revision=null`; its original base revision is unproven.
        This job explicitly loads Qwen revision `b968826d9c46dd6066d109eabc6255188de91218`.

        Cells 1–4 install/verify dependencies and load only a tokenizer. Cell 5 requires typed authorization
        before any full-model loading or training. Cell 6 exports results or failure diagnostics.
        There is no inference/model-quality evaluation or checkpoint promotion beyond one nonempty-generation smoke test.
    ''')
    md('## 1. Initialize the transfer workspace and embedded runner')
    code('''
        import os, sys, json, subprocess, shutil, hashlib, platform
        from pathlib import Path
        if platform.system() != 'Linux' or platform.machine() != 'x86_64' or not (3, 10) <= sys.version_info[:2] <= (3, 12):
            raise RuntimeError('Use a Linux x86-64 Python 3.10–3.12 notebook runtime; do not change dependencies silently.')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        BASE = Path('/kaggle/working/test-b') if Path('/kaggle/working').exists() else Path('/content/test-b')
        BASE.mkdir(parents=True, exist_ok=True)
        ROOT = BASE / 'package'
        RUNNER = BASE / 'gpu_canary_runner.py'
        VENV = BASE / 'venv'
        PY = VENV / 'bin/python'
        os.environ['HF_HOME'] = str(BASE / 'hf-cache')

        def execute(command, log_name):
            log = BASE / log_name
            if log.exists():
                raise RuntimeError(f'Log already exists: {log}. Inspect earlier attempt before rerunning.')
            with log.open('w') as stream:
                process = subprocess.Popen([str(x) for x in command], cwd=BASE, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                try:
                    for line in process.stdout:
                        stream.write(line); stream.flush(); print(line, end='')
                    returncode = process.wait()
                except BaseException:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
                    raise
            if returncode:
                raise RuntimeError(f'Command failed ({returncode}); inspect {log}. Do not continue to training.')
    ''')
    code('RUNNER.write_text(' + repr(helper) + ')\nassert hashlib.sha256(RUNNER.read_bytes()).hexdigest() == ' + repr(hashlib.sha256(helper.encode()).hexdigest()) + '\nprint("Runner ready; no training started.")')
    md('''
        ## 2. Upload and verify the existing package
        Colab will open a file picker: choose the original `package.zip` from the SSD.
        Kaggle: attach the ZIP as an input dataset, then paste its `/kaggle/input/.../package.zip` path.
        Keep this project’s adapter/data private when creating the input dataset.
    ''')
    code('''
        if Path('/content').exists() and not Path('/kaggle/working').exists():
            from google.colab import files
            previous = Path.cwd()
            os.chdir(BASE)
            try: uploaded = files.upload()
            finally: os.chdir(previous)
            if len(uploaded) != 1:
                raise RuntimeError('Upload exactly the original package ZIP.')
            ARCHIVE = BASE / next(iter(uploaded))
            del uploaded
        else:
            ARCHIVE = Path(input('Uploaded original package ZIP path: ').strip())
        execute([sys.executable, RUNNER, 'extract', '--archive', ARCHIVE, '--root', ROOT], 'extract.log')
        print('Original package hash and every source, dataset and adapter hash verified.')
    ''')
    md('''
        ## 3. Install isolated CUDA/PyTorch and packaged project dependencies
        This does not load model weights. Allow at least 40 GiB free disk for dependencies, model cache and outputs.
        The CUDA wheel needs a compatible host NVIDIA driver. Driver failures must be fixed before training.
        Pinning is based on the [official PyTorch 2.8 CUDA 12.6 wheel instructions](https://pytorch.org/get-started/previous-versions/)
        and [bitsandbytes 0.48.2 compatibility](https://huggingface.co/docs/bitsandbytes/v0.48.2/installation).
        The complete combination still requires live validation on your allocated GPU.
    ''')
    code('''
        if shutil.disk_usage(BASE).free < 40 * 1024**3:
            raise RuntimeError('Need at least 40 GiB free cloud-local disk; no model has been loaded.')
        if VENV.exists():
            raise RuntimeError('Environment already exists; inspect installation logs instead of reinstalling automatically.')
        execute([sys.executable, '-m', 'venv', VENV], 'venv.log')
        constraints = BASE / 'constraints.txt'
        constraints.write_text('torch==2.8.0\\ntorchvision==0.23.0\\ntorchaudio==2.8.0\\ntransformers==4.57.3\\ntrl==0.25.1\\npeft==0.18.0\\nlighteval==0.11.0\\naccelerate==1.11.0\\nbitsandbytes==0.48.2\\ndatasets==4.4.1\\ntokenizers==0.22.2\\nhuggingface-hub==0.36.0\\nsafetensors==0.6.2\\n')
        execute([PY, '-m', 'pip', 'install', '--upgrade', 'pip'], 'pip-bootstrap.log')
        execute([PY, '-m', 'pip', 'install', '--no-cache-dir', 'torch==2.8.0', 'torchvision==0.23.0',
                 'torchaudio==2.8.0', '--index-url', 'https://download.pytorch.org/whl/cu126'], 'pytorch-install.log')
        execute([PY, '-m', 'pip', 'install', '--no-cache-dir', '-c', constraints,
                 '-r', ROOT / 'requirements-training.txt', 'tokenizers==0.22.2',
                 'huggingface-hub==0.36.0', 'safetensors==0.6.2'], 'dependencies-install.log')
        execute([PY, '-m', 'pip', 'check'], 'pip-check.log')
        execute([PY, '-m', 'pip', 'freeze'], 'pip-freeze.log')
    ''')
    md('''
        ## 4. Check GPU and run the original worker’s validate-only mode
        Verifies package integrity, all 85 targets, tokenizer/context, non-thinking template,
        actual TRL completion-only masks and zero truncation. No full model loading or training.
        Inspect `cloud-preflight.json` and both validate-only logs if anything fails.
    ''')
    code('''
        execute([PY, RUNNER, 'preflight', '--root', ROOT], 'preflight.log')
        PREFLIGHT = json.loads((BASE / 'cloud-preflight.json').read_text())
        GPU = PREFLIGHT['environment']['gpu']
        print('Detected GPU:', GPU)
        print('Maximum sequence:', PREFLIGHT['tokenizer']['sequence_tokens']['max'])
        print('Overlength:', PREFLIGHT['tokenizer']['overlength_examples'])
        print('STOP HERE until you approve the actual allocation and two-step run.')
    ''')
    md('''
        ## 5. Explicitly authorize this allocation and run exactly two steps
        Do not run this cell until preflight passes. Enter the actual provider and billing status.
        For free Colab/Kaggle, confirm the specific allocation costs $0; no invented comparison hourly rate is used.
        Paid mode requires the actual approved positive hourly rate and a budget no greater than $1.
        Colab compute-unit pricing that cannot be expressed as a known USD/hour rate is not supported in paid mode.
        The worker watchdog covers this worker only; it cannot stop provider idle charges. Shut down paid hardware yourself.
        An additional 1,200-second cap can only tighten the worker limit. No retries, step/context increases or new adapters.
        The wrapper records actual steps/losses, 4-bit loading, exact adapter restores, inference text and peak PyTorch VRAM.
        Free mode calls the unchanged worker core with an explicit wall-clock watchdog because the packaged paid CLI rejects rate zero.
    ''')
    code('''
        if (ROOT / 'result').exists() or (BASE / 'training-observations.json').exists():
            raise RuntimeError('Previous training attempt exists. Export it; do not retrain automatically.')
        provider = input('Actual GPU provider (Colab/Kaggle/other approved provider): ').strip()
        billing = input('Actual billing mode: type free or paid: ').strip().lower()
        if billing not in ('free', 'paid'): raise RuntimeError('Unknown billing mode')
        if billing == 'free':
            if input('Type FREE to confirm this allocation has no usage charge: ').strip() != 'FREE':
                raise RuntimeError('Free allocation not confirmed')
            rate, budget = '0', '1'
        else:
            rate = input('Actual approved GPU hourly rate in USD: ').strip()
            budget = input('Approved worker budget in USD, at most 1: ').strip()
        statement = f'APPROVE {provider} | {GPU} | {rate} USD/hour | {budget} USD cap | 2 steps'
        print('Review and type exactly:', statement)
        if input().strip() != statement:
            raise RuntimeError('Training not authorized; no model loaded')
        execute([PY, RUNNER, 'train', '--root', ROOT, '--authorize-training', '--provider', provider,
                 '--expected-gpu', GPU, '--billing-mode', billing, '--gpu-hourly-rate-usd', rate,
                 '--max-gpu-usd', budget, '--max-runtime-seconds', '1200'], 'training.log')
        outcome = json.loads((ROOT / 'result/result.json').read_text())
        print('Runtime seconds:', outcome['gpu_runtime_seconds'])
        print('GPU cost USD:', outcome['estimated_gpu_cost_usd'])
        print('Observations:', json.dumps(outcome['canary_observations'], indent=2))
    ''')
    md('''
        ## 6. Export success results or exact failure diagnostics
        Run this cell after success OR failure. Download the ZIP and SHA-256 sidecar before shutting down.
        A failed run is diagnostic only and must not be imported as a completed checkpoint.
    ''')
    code('''
        execute([PY if PY.exists() else sys.executable, RUNNER, 'export', '--root', ROOT], 'export.log')
        archive = BASE / 'test-b-return.zip'
        sidecar = BASE / 'test-b-return.zip.sha256'
        print(sidecar.read_text())
        if Path('/content').exists() and not Path('/kaggle/working').exists():
            from google.colab import files
            files.download(str(archive)); files.download(str(sidecar))
        else:
            from IPython.display import FileLink, display
            display(FileLink(str(archive))); display(FileLink(str(sidecar)))
            print('Also available under Kaggle /kaggle/working/test-b/. Download before ending the session.')
    ''')
    md('''
        ## Return to the Mac
        Save both downloads on the external SSD under `experiments/gpu-canary-20261119/test-b/returned/`.
        Verify the ZIP hash against the sidecar, extract it, and inspect `result/result.json` and
        `diagnostics/training-observations.json` before import. The accompanying runbook gives exact Mac commands.
        Preserve the original adapter. Do not promote this two-step engineering checkpoint or claim model improvement.
    ''')
    output = ROOT / 'notebooks/qwen3_8b_continued_gpu_canary.ipynb'
    for index, cell in enumerate(cells):
        cell['id'] = hashlib.sha256((str(index) + cell['source']).encode()).hexdigest()[:12]
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'},
        'colab': {'name': output.name}, 'accelerator': 'GPU'}, 'cells': cells}, indent=2) + '\n')
    return output


if __name__ == '__main__':
    print(build())
