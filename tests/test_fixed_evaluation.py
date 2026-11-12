import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from distillation_loop.benchmarks import load_benchmarks
from distillation_loop.build_fixed_evaluation import build_rows, OUTPUT
from distillation_loop.controller import run_loop
from distillation_loop.core import load_loop_config
from distillation_loop.preflight import readiness
from distillation_loop.smoke import smoke_configs
from self_instruct_pipeline.config import load_config, ROOT
from self_instruct_pipeline.persistence import atomic_json, atomic_jsonl, digest
from self_instruct_pipeline.validation import validate_answer
import test_distillation_loop as fixtures


class FixedEvaluationTests(unittest.TestCase):
    def test_checked_in_corpus_reproduces_and_validates_all_540(self):
        config=load_loop_config()
        self.assertEqual(config['evaluation']['mode'],'fixed')
        config['evaluation'].pop('holdout_families_per_task')
        config['development_examples_per_task'] = 30
        rows,test=load_benchmarks(config,fixtures.SCHEMAS)
        self.assertEqual(test,[])
        self.assertEqual(len(rows),540)
        self.assertEqual(Counter(x['record']['task'] for x in rows),Counter({t:30 for t in fixtures.SCHEMAS}))
        self.assertEqual(digest(rows),digest(build_rows()))
        for item in rows:
            self.assertFalse(item['verification']['human_reviewed'])
            validate_answer(item['record'],item['reference'],fixtures.SCHEMAS[item['record']['task']])
            # References quote only applicable original sources, not the distractions.
            for evidence in item['reference'].get('evidence',[]):
                if isinstance(evidence,dict):self.assertNotIn(evidence['document_id'],('INJECTION','UNRELATED'))

    def test_missing_or_tampered_file_never_falls_back_to_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'evaluation.jsonl';manifest=Path(folder)/'evaluation.manifest.json'
            config=load_loop_config()
            config['evaluation'].update(validation_path=str(path),manifest_path=str(manifest))
            with patch('distillation_loop.data.Pipeline',side_effect=AssertionError('No API generation')):
                report=readiness(load_config(),config,check_runtime=False)
                self.assertFalse(report['ready'])
                self.assertIn('No API generation',report['issues'][0])
                path.write_text(OUTPUT.read_text());manifest.write_text(OUTPUT.with_suffix('.manifest.json').read_text())
                rows=[json.loads(line) for line in path.read_text().splitlines()]
                rows[0]['reference']['answer']='Changed answer'
                atomic_jsonl(path,rows)
                with self.assertRaisesRegex(ValueError,'hash has changed'):load_benchmarks(config,fixtures.SCHEMAS)

    def test_preflight_and_smoke_load_local_data_without_api_calls(self):
        with patch('distillation_loop.data.Pipeline',side_effect=AssertionError('No API generation')):
            report=readiness(load_config(),load_loop_config(),check_runtime=False)
            self.assertTrue(report['ready'])
            self.assertEqual(report['validation_examples'],432)
            self.assertEqual(report['test_examples'],108)
            self.assertEqual(report['generated_validation_target'],0)
            _,smoke=smoke_configs(load_config(),load_loop_config(),'/tmp/fixed-smoke-fixture')
            rows,_=load_benchmarks(smoke,fixtures.SCHEMAS)
            self.assertEqual(len(rows),18)
            self.assertEqual(Counter(x['record']['task'] for x in rows),Counter({t:1 for t in fixtures.SCHEMAS}))

    def test_controller_uses_bundled_cases_instead_of_teacher_development_generation(self):
        class Evaluator:
            def evaluate(self,rows,directory):
                if not hasattr(self, 'rows'):
                    self.rows = rows
                report=fixtures.report(.99)
                report['examples_per_task']={t:30 for t in fixtures.SCHEMAS}
                return report
        evaluator=Evaluator()
        with tempfile.TemporaryDirectory() as folder:
            config=load_loop_config();config['output_dir']=folder
            with patch('distillation_loop.controller.frozen_development',side_effect=AssertionError('No API evaluation generation')),patch('distillation_loop.controller.LightevalEvaluator',return_value=evaluator):
                result=run_loop(fixtures.FakeRuntime(),load_config(),config)
            self.assertEqual(len(evaluator.rows),432)
            self.assertFalse(result['promotion']['accepted'])
            self.assertIn('final_comparison', result)
            split = json.loads((Path(folder) / 'evaluation_split.json').read_text())
            self.assertFalse(set(split['development']['source_groups']) & set(split['holdout']['source_groups']))
            self.assertEqual(result['completed_rounds'],0)
            self.assertEqual(result['development_hash'],digest(evaluator.rows))

    def test_numeric_and_logic_edge_case_reference_answers(self):
        rows=build_rows()
        by_id={x['record']['id']:x for x in rows}
        def answer(task,k,v=0):return by_id[f'fixed-synthetic-v1-{task}-{k:02d}-{v}']['reference']
        self.assertIn('USD -25.00',answer('patient_billing',2)['answer'])
        self.assertIn('USD 0.00',answer('patient_billing',1)['answer'])
        self.assertIn('not met',answer('medical_necessity_support',0)['answer'])
        self.assertIn('criteria are met',answer('medical_necessity_support',1)['answer'])
        self.assertIn('2024-03-06',answer('appeals_process_qa',0)['answer'])
        self.assertIn('2024-03-05',answer('appeals_process_qa',1)['answer'])
        self.assertEqual(answer('clinical_coding',7)['suggested_codes'],['TR-B0'])
