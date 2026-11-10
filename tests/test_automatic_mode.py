import json
import unittest
from pathlib import Path

from distillation_loop.benchmarks import load_benchmarks
from distillation_loop.core import load_loop_config
from distillation_loop.preflight import readiness
from self_instruct_pipeline.config import load_config
import test_distillation_loop as fixtures


class AutomaticModeTests(unittest.TestCase):
    def test_full_default_uses_automatic_checks_without_external_benchmarks(self):
        config=load_loop_config()
        self.assertEqual(config['evaluation']['mode'],'automatic')
        self.assertEqual(config['max_rounds'],5)
        config['evaluation']['validation_path']='/nonexistent/validation.jsonl'
        config['evaluation']['test_path']='/nonexistent/test.jsonl'
        report=readiness(load_config(),config,check_runtime=False)
        self.assertTrue(report['ready'])
        self.assertFalse(report['manual_review_required'])
        self.assertEqual(report['generated_validation_target'],540)
        self.assertEqual(load_benchmarks(config,fixtures.SCHEMAS),(None,[]))

    def test_full_automatic_training_selects_checkpoint_without_claiming_human_review(self):
        f=fixtures.ControllerTests();f.setUp();self.addCleanup(f.tmp.cleanup)
        f.config['evaluation']={'mode':'automatic'}
        f.config['max_rounds']=1
        runtime=fixtures.FakeRuntime()
        result=f.run_loop(runtime,fixtures.FakeEvaluator([.1,.95]))
        self.assertEqual(len(runtime.trains),1)
        self.assertEqual(result['evaluation_status'],'automatically_evaluated')
        self.assertEqual(result['selected_checkpoint'],result['best_checkpoint'])
        self.assertFalse(result['promotion']['accepted'])
        self.assertEqual(runtime.restores[-1],result['best_checkpoint'])

    def test_notebooks_default_to_full_automatic_mode_and_have_matching_cells(self):
        books=[json.loads(Path(name).read_text()) for name in ('distillation.ipynb','notebook77892919a9_fixed.ipynb')]
        self.assertEqual([c['source'] for c in books[0]['cells']],[c['source'] for c in books[1]['cells']])
        settings=next(''.join(c['source']) for c in books[0]['cells'] if c['cell_type']=='code' and 'TEST_RUN = False' in ''.join(c.get('source',[])))
        self.assertIn('loop_config["evaluation"]["mode"] = "automatic"',settings)
        self.assertIn('distillation_loop_automatic_v1',settings)
