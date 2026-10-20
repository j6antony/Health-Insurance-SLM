import time
import unittest
from unittest.mock import patch
import pymupdf
from fastapi.testclient import TestClient
from src.web import server


class WebTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)

    def pdf(self):
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((40, 40), 'Synthetic test document')
            return pdf.tobytes()

    def wait(self, job_id):
        for _ in range(100):
            job = self.client.get('/api/jobs/' + job_id).json()
            if job['status'] != 'processing':
                return job
            time.sleep(.01)
        self.fail('Worker did not finish')

    def test_health_and_invalid_upload(self):
        self.assertEqual(self.client.get('/api/health').status_code, 200)
        self.assertEqual(self.client.post('/api/verify').status_code, 422)
        self.assertEqual(self.client.post('/api/verify', files={'files':('x.pdf',b'invalid','application/pdf')}).status_code, 400)
        self.assertEqual(self.client.post('/api/verify', files={'files':('x.txt',self.pdf(),'text/plain')}).status_code, 400)
        self.assertEqual(self.client.get('/api/jobs/missing').status_code, 404)

    def test_pipeline_and_cleanup(self):
        paths = []
        def process(files):
            paths.extend(files)
            self.assertTrue(files[0].exists())
            return {'documents':[{'filename':files[0].name}], 'requirements':{'satisfied':False}}
        with patch.object(server,'process_claim_package',side_effect=process):
            response = self.client.post('/api/verify',files={'files':('claim.pdf',self.pdf(),'application/pdf')})
            self.assertEqual(response.status_code,202)
            job = self.wait(response.json()['job_id'])
            self.assertEqual(job['status'],'complete')
            self.assertEqual(job['result']['documents'][0]['filename'],'claim.pdf')
            self.assertFalse(paths[0].exists())

    def test_failure_and_busy(self):
        with patch.object(server,'process_claim_package',side_effect=ConnectionError('offline')):
            response = self.client.post('/api/verify',files={'files':('claim.pdf',self.pdf(),'application/pdf')})
            job = self.wait(response.json()['job_id'])
            self.assertEqual(job['status'],'failed')
            self.assertIn('Ollama',job['error'])
        server.BUSY.acquire()
        try:
            response = self.client.post('/api/verify',files={'files':('claim.pdf',self.pdf(),'application/pdf')})
            self.assertEqual(response.status_code,409)
        finally:
            server.BUSY.release()
