import unittest
import tempfile
import json
from unittest.mock import patch
from pathlib import Path

from src.preprocessing.document_builder import process_document
from src.preprocessing.section_detector import segment_with_slm
from src.preprocessing.section_chunker import chunk_sections

ROOT = Path(__file__).resolve().parents[2]


class SectioningRegression(unittest.TestCase):
    def segment(self, lines):
        # Deliberately bad model output: every line is a heading.
        return segment_with_slm(lines, [
            {'line_id': i, 'label': 'section_heading'}
            for i in range(1, len(lines) + 1)
        ])

    def test_false_headings(self):
        texts = ['Hospital information', 'UB-04 form', 'OR',
                 'Completed physician statement', 'OR', 'All of the below:',
                 '• Required documents', 'and a disclosure authorization',
                 'Admission Date and Time*', 'Number of Days in ICU', 'Treatment Period:']
        lines = [{'text': t, 'is_bold': True} for t in texts]
        sections = self.segment(lines)
        self.assertEqual([s['heading'] for s in sections], ['Hospital information'])
        chunks = chunk_sections(sections, max_words=5)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0]['oversized'])
        self.assertEqual(chunks[0]['word_count'], len(chunks[0]['text'].split()))
        self.assertEqual(chunks[0]['source_lines'], list(range(2, len(lines) + 1)))

    def test_heading_only_and_validation(self):
        sections = self.segment([{'text': 'History', 'is_bold': True}])
        self.assertEqual(chunk_sections(sections)[0]['text'], 'History\n')
        with self.assertRaises(ValueError):
            chunk_sections(sections, 0)
        with self.assertRaises(ValueError):
            segment_with_slm([{'text': 'x'}], [])

    def test_actual_checklist_columns_and_alternatives(self):
        page = process_document(ROOT / 'data/raw/reference_pdfs/cigna_hospital_care_checklist.pdf')['pages'][0]
        lines = page['lines']
        texts = [l['text'] for l in lines]
        employee = texts.index('Employee Basic Information')
        hospital = texts.index('Hospital information')
        self.assertLess(employee, texts.index('Home address'))
        self.assertLess(texts.index('Home address'), hospital)
        self.assertLess(texts.index('OR'), texts.index('Please note:'))
        sections = self.segment(lines)
        hospital_section = next(s for s in sections if s['heading'] == 'Hospital information')
        content = hospital_section['content']
        self.assertEqual(content.count('OR'), 2)
        self.assertIn('Completed physician statement', content)
        chunks = chunk_sections([hospital_section], 10)
        self.assertEqual(len(chunks), 1)
        self.assertIn('Completed physician statement', chunks[0]['text'])
        self.assertFalse(any((s['heading'] or '').startswith('and ') for s in sections))
        ids = [s['heading_line_id'] for s in sections if s['heading_line_id']]
        ids += [l['line_id'] for s in sections for l in s['source_lines']]
        self.assertEqual(sorted(ids), list(range(1, len(lines) + 1)))

    def test_pipeline_output(self):
        from src.preprocessing.pipeline import preprocess_document

        def classify(lines):
            return {'predictions': [{'line_id': i, 'label': 'section_heading'}
                                    for i in range(1, len(lines) + 1)]}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'document.json'
            with patch('src.preprocessing.pipeline.classify_document_lines', side_effect=classify):
                document = preprocess_document(
                    ROOT / 'data/raw/reference_pdfs/cigna_hospital_care_checklist.pdf',
                    output, max_words=10)
            self.assertEqual(json.loads(output.read_text()), document)
            self.assertEqual(document['chunk_count'], len(document['chunks']))
            self.assertEqual(len({c['chunk_id'] for c in document['chunks']}), document['chunk_count'])
            for chunk in document['chunks']:
                self.assertEqual(chunk['word_count'], len(chunk['text'].split()))
                self.assertEqual(chunk['document_id'], document['document_id'])

    def test_paragraph_continuation_stays_together(self):
        lines = [{'text': '• Provide the death certificate', 'line_id': [1, 0, 0]},
                 {'text': 'and a disclosure authorization', 'line_id': [1, 0, 1]},
                 {'text': 'Resources department.', 'line_id': [1, 0, 2]}]
        sections = self.segment(lines)
        chunks = chunk_sections(sections, 3)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0]['oversized'])

    def test_actual_form_reading_order(self):
        page = process_document(ROOT / 'data/raw/reference_pdfs/cigna_hospital_care_claim_form.pdf')['pages'][1]
        lines = page['lines']
        headings = [l['text'] for l in lines if l['text'].startswith('SECTION ')]
        self.assertEqual([h.split(':')[0] for h in headings],
                         [f'SECTION {i}' for i in range(1, 6)])
        sections = segment_with_slm(lines, [
            {'line_id': i, 'label': 'not_heading'} for i in range(1, len(lines) + 1)
        ])
        self.assertEqual([s['heading'] for s in sections if s['heading']], headings)
        for section in sections:
            if section['heading'] and section['heading'].startswith('SECTION 1:'):
                self.assertTrue(any('Date of Birth' in t for t in section['content']))
                self.assertFalse(any('Is the Child' in t for t in section['content']))


if __name__ == '__main__':
    unittest.main()
