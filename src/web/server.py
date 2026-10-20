"""FastAPI backend for the React claim verification frontend."""
import logging
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.pipeline.claim_pipeline import process_claim_package

app = FastAPI(title='Claim verification API')
MAX_BYTES = 30 * 1024 * 1024
JOBS = {}
LOCK = threading.Lock()
BUSY = threading.Semaphore(1)


def run_job(job_id, uploads):
    try:
        with tempfile.TemporaryDirectory(prefix='claim-upload-') as directory:
            paths, names = [], {}
            for index, (name, data) in enumerate(uploads):
                path = Path(directory) / f'{index + 1}.pdf'
                path.write_bytes(data)
                with pymupdf.open(path) as pdf:
                    if pdf.is_encrypted or not 1 <= len(pdf) <= 50:
                        raise ValueError('PDFs must be unlocked and contain 1–50 pages.')
                paths.append(path)
                names[path.name] = name
            result = process_claim_package(paths)
            for document in result['documents']:
                document['filename'] = names.get(document['filename'], document['filename'])
            with LOCK:
                JOBS[job_id].update(status='complete', result=result)
    except Exception:
        logging.exception('Claim processing failed')
        with LOCK:
            JOBS[job_id].update(status='failed', error='Verification failed. Check that PDFs are readable and unlocked, and Ollama is running with qwen3:4b-instruct. See the backend terminal for details.')
    finally:
        BUSY.release()


@app.get('/api/health')
def health():
    return {'status': 'ok'}


@app.post('/api/verify', status_code=202)
async def verify(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= 10:
        raise HTTPException(400, 'Choose 1–10 PDF files.')
    uploads, total = [], 0
    try:
        for file in files:
            name = Path(file.filename or '').name
            if not name.lower().endswith('.pdf'):
                raise HTTPException(400, 'Only PDF files are supported.')
            data = await file.read(MAX_BYTES - total + 1)
            total += len(data)
            if not data or total > MAX_BYTES:
                raise HTTPException(400, 'Choose nonempty PDFs totaling at most 30 MB.')
            if b'%PDF-' not in data[:1024]:
                raise HTTPException(400, f'{name} is not a PDF.')
            uploads.append((name[:200], data))
    finally:
        for file in files:
            await file.close()
    if not BUSY.acquire(blocking=False):
        raise HTTPException(409, 'Another package is processing. Please wait and retry.')
    job_id = uuid.uuid4().hex
    with LOCK:
        for key in list(JOBS):
            if time.time() - JOBS[key]['created_at'] > 3600:
                del JOBS[key]
        while len(JOBS) >= 10:
            del JOBS[next(iter(JOBS))]
        JOBS[job_id] = {'status': 'processing', 'created_at': time.time()}
    try:
        threading.Thread(target=run_job, args=(job_id, uploads), daemon=True).start()
    except Exception:
        BUSY.release()
        with LOCK:
            JOBS.pop(job_id, None)
        raise
    return {'job_id': job_id}


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, 'Result not found. Please upload again.')
        return dict(job)
