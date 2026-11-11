import ast
import copy
import hashlib
import json
import os
import re
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from distillation_loop.colab import colab_configs, save_colab_subset
from distillation_loop.benchmarks import load_benchmarks
from distillation_loop.core import load_loop_config
from distillation_loop.drive_backup import save_drive_adapter
from distillation_loop.student import StudentRuntime
from self_instruct_pipeline.config import load_config, ROOT
from self_instruct_pipeline.persistence import Store, digest
import test_distillation_loop as fixtures


class ColabProfileTests(unittest.TestCase):
    def profile(self,run_dir='/content/drive/MyDrive/HealthInsuranceSLM/colab_fixed100_v1'):
        return colab_configs(load_config(),load_loop_config(),run_dir)

    def test_profile_is_isolated_and_loads_exactly_100_cases(self):
        parent, original=load_config(),load_loop_config()
        before=digest([parent,original])
        _,profile=colab_configs(parent,original,'/tmp/test-colab')
        self.assertEqual(digest([parent,original]),before)
        rows,_=load_benchmarks(profile,fixtures.SCHEMAS)
        counts=Counter(x['record']['task'] for x in rows)
        self.assertEqual(len(rows),100)
        self.assertEqual(len({x['record']['id'] for x in rows}),100)
        self.assertEqual(set(counts),set(fixtures.SCHEMAS))
        self.assertEqual(Counter(counts.values()),Counter({5:8,6:10}))
        for task in counts:
            self.assertEqual(len({x['verification']['family'] for x in rows if x['record']['task']==task}),counts[task])
        self.assertEqual(profile['max_rounds'],3)
        self.assertEqual(profile['scenarios_per_round'],24)
        self.assertEqual(profile['inference']['batch_size'],1)
        self.assertEqual(profile['training']['lora_rank'],8)
        # Kaggle's default still loads its complete 540 examples.
        self.assertEqual(len(load_benchmarks(original,fixtures.SCHEMAS)[0]),540)

    def test_subset_stability_limits_and_export_resume(self):
        _,profile=self.profile()
        rows,_=load_benchmarks(profile,fixtures.SCHEMAS)
        self.assertEqual(digest(rows),digest(load_benchmarks(profile,fixtures.SCHEMAS)[0]))
        changed=copy.deepcopy(profile);changed['evaluation']['subset_seed']=43
        self.assertNotEqual(digest(rows),digest(load_benchmarks(changed,fixtures.SCHEMAS)[0]))
        for size in (89,541,100.5,True):
            bad=copy.deepcopy(profile);bad['evaluation']['total_examples']=size
            with self.assertRaises(ValueError):load_benchmarks(bad,fixtures.SCHEMAS)
        with tempfile.TemporaryDirectory() as folder:
            one=save_colab_subset(profile,fixtures.SCHEMAS,folder)
            self.assertEqual(one,save_colab_subset(profile,fixtures.SCHEMAS,folder))
            with self.assertRaisesRegex(ValueError,'subset changed'):save_colab_subset(changed,fixtures.SCHEMAS,folder)
            path=Path(folder)/'evaluation_100.jsonl';path.write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'cases have changed'):save_colab_subset(profile,fixtures.SCHEMAS,folder)

    def test_single_gpu_runtime_respects_inference_batch_size_one(self):
        runtime=object.__new__(StudentRuntime)
        runtime.config={'training':{'max_sequence_length':100},'inference':{'max_new_tokens':10,'batch_size':1}}
        runtime.prepare=lambda:None
        runtime.tokenizer=lambda *a,**k:{'input_ids':[1,2]}
        batches=[]
        def generate(texts):
            batches.append(len(texts))
            return [{'raw':'{}','truncated':False} for text in texts]
        runtime._generate_batch=generate
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=type('OOM',(Exception,),{})))
        with patch.dict('sys.modules',{'torch':fake_torch}),patch('distillation_loop.student.render_prompt',return_value='prompt'):
            rows=runtime.infer_many([{}, {}, {}])
        self.assertEqual(len(rows),3)
        self.assertEqual(batches,[1,1,1])

    def test_drive_data_uses_local_process_locks(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'drive/run';locks=root/'local-locks'
            with patch.dict('os.environ',{'DISTILLATION_LOCAL_LOCK_DIR':str(locks)}):
                first,second=Store(data),Store(root/'drive/other')
                with first.lock():
                    with self.assertRaisesRegex(ValueError,'Another pipeline'): 
                        with Store(data).lock():pass
                    with second.lock():pass
                    first.append('journal.jsonl',{'saved':True})
                with first.lock():pass
            self.assertFalse((data/'.pipeline.lock').exists())
            self.assertEqual(len(list(locks.glob('*.lock'))),2)
            self.assertEqual(first.read('journal.jsonl'),[{'saved':True}])


class DriveBackupTests(unittest.TestCase):
    def adapter(self,folder):
        adapter=Path(folder)/'round_01/sft/adapter';adapter.mkdir(parents=True)
        (adapter/'adapter_config.json').write_text('{}')
        (adapter/'adapter_model.safetensors').write_bytes(b'weights')
        (adapter/'tokenizer.json').write_text('{}')
        (adapter/'.env').write_text('SECRET')
        return adapter

    def test_archive_integrity_allowlist_and_repeated_export(self):
        with tempfile.TemporaryDirectory() as folder:
            adapter=self.adapter(folder)
            one=save_drive_adapter(adapter,folder,1)
            archive=Path(one['archive']);before=archive.stat().st_mtime_ns
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(set(z.namelist()),{'adapter_config.json','adapter_model.safetensors','tokenizer.json','manifest.json'})
            self.assertEqual(one,save_drive_adapter(adapter,folder,1))
            self.assertEqual(archive.stat().st_mtime_ns,before)
            self.assertEqual(one['sha256'],hashlib.sha256(archive.read_bytes()).hexdigest())
            archive.write_bytes(b'corrupt backup')
            repaired=save_drive_adapter(adapter,folder,1)
            self.assertTrue(zipfile.is_zipfile(repaired['archive']))
            self.assertEqual(repaired['sha256'],one['sha256'])

    def test_backup_failure_reuses_finished_training_on_resume(self):
        f=fixtures.ControllerTests();f.setUp();self.addCleanup(f.tmp.cleanup)
        f.config['max_rounds']=1
        runtime=fixtures.FakeRuntime();evaluator=fixtures.FakeEvaluator([.1,.95])
        with patch.dict('os.environ',{'DISTILLATION_DRIVE_BACKUP':'1'}):
            with patch('distillation_loop.drive_backup.save_drive_adapter',side_effect=OSError('Drive unavailable')):
                with self.assertRaises(OSError):f.run_loop(runtime,evaluator)
            self.assertEqual(len(runtime.trains),1)
            with patch('distillation_loop.drive_backup.save_drive_adapter',return_value={'status':'drive_archive_verified'}) as saved:
                result=f.run_loop(runtime,evaluator)
                saved.assert_called_once()
        self.assertEqual(result['completed_rounds'],1)
        self.assertEqual(len(runtime.trains),1)

    def test_subset_training_excludes_all_540_cases(self):
        _,profile=colab_configs(load_config(),load_loop_config(),'/tmp/placeholder')
        class Evaluator:
            def __init__(self):self.calls=0
            def evaluate(self,rows,directory):
                self.calls+=1
                counts=Counter(x['record']['task'] for x in rows)
                report=fixtures.report(.1 if self.calls==1 else .95)
                report['examples_per_task']=dict(counts)
                return report
        with tempfile.TemporaryDirectory() as folder:
            profile.update(output_dir=folder,max_rounds=1)
            def curriculum(base,excluded,directory):
                self.assertEqual(len(excluded),540)
                return fixtures.FakeCurriculum(base,excluded,directory)
            profile['quality']['minimum_deterministic_check_rate']=0
            evaluator=Evaluator()
            from distillation_loop.controller import run_loop
            with patch('distillation_loop.controller.LightevalEvaluator',return_value=evaluator),patch('distillation_loop.controller.CurriculumData',side_effect=curriculum):
                result=run_loop(fixtures.FakeRuntime(),load_config(),profile)
            self.assertEqual(result['completed_rounds'],1)


class ColabNotebookTests(unittest.TestCase):
    def test_notebook_cells_parse_and_use_direct_single_gpu_runtime(self):
        nb=json.loads((ROOT/'distillation_colab_100.ipynb').read_text())
        code=''
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                text=''.join(cell['source']);code+=text
                ast.parse(''.join('pass\n' if line.startswith('%') else line for line in cell['source']))
        self.assertIn('StudentRuntime(model, tokenizer, loop_config)',code)
        self.assertNotIn('DualGPURuntime',code)
        self.assertIn('device_map={"": 0}',code)
        self.assertIn('drive.mount',code)
        self.assertIn('RUN_DIR = DRIVE_ROOT / RUN_NAME',code)
        self.assertIn('os.environ["GITHUB_ADAPTER_UPLOAD"] = "0"',code)

    def test_portable_bundle_matches_notebook_and_contains_no_run_or_secrets(self):
        from tools.build_colab_bundle import build
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        bundle=build(Path(temp.name)/'colab_distillation_bundle.zip',update_notebook=False)
        nb=json.loads((ROOT/'distillation_colab_100.ipynb').read_text())
        code='\n'.join(''.join(c.get('source',[])) for c in nb['cells'])
        expected=re.search(r'EXPECTED_BUNDLE_SHA256 = "([0-9a-f]+)"',code).group(1)
        self.assertEqual(hashlib.sha256(bundle.read_bytes()).hexdigest(),expected)
        with zipfile.ZipFile(bundle) as z:
            names=z.namelist()
            self.assertIn('Health-Insurance-SLM-Colab/distillation_loop/colab.py',names)
            self.assertIn('Health-Insurance-SLM-Colab/data/evaluation/fixed_validation_v1.jsonl',names)
            self.assertFalse(any('/outputs/' in name or '/.git/' in name or '/.env' in name or name.endswith('.safetensors') for name in names))
