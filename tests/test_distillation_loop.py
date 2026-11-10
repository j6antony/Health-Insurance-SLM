import copy
import json
import math
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from distillation_loop.controller import run_loop
from distillation_loop.core import (ResponseError, aggregate_scores, clean_response, feedback_weights,
                                    load_loop_config, satisfactory, score_response)
from distillation_loop.student import tokenize_teacher_pair
from self_instruct_pipeline.config import ROOT, load_config
from self_instruct_pipeline.persistence import atomic_json, digest, read_jsonl

SCHEMAS=json.loads((ROOT/'data/self_instruct/task_schemas.json').read_text())['tasks']


def record(task='document_classification', i=0):
    return {'id':f'case-{i}','task':task,'instruction':'Classify the supplied document using the given schema.',
            'scenario':{'synthetic':True,'documents':[{'document_id':'D1','page_number':1,
                        'text':f'Synthetic hospital invoice {i}. Total USD 50.'}],'context':{}}}


def report(score,valid=1.0):
    return {'macro_task_score':score,'minimum_task_score':score,
            'per_task':{t:score for t in SCHEMAS},'schema_evidence_valid_rate':valid,
            'examples_per_task':{t:3 for t in SCHEMAS}}


class FakeTokenizer:
    eos_token_id=9
    def apply_chat_template(self,messages,**kwargs):
        if kwargs['enable_thinking'] is not False:raise AssertionError('thinking enabled')
        return 'PROMPT'
    def __call__(self,text,**kwargs):
        return {'input_ids':[1,2,3] if text=='PROMPT' else [4,5]}


class FakeRuntime:
    def __init__(self,crash_train=False):
        self.trains=[];self.restores=[];self.inferences=[];self.crash_train=crash_train
    def prepare(self):pass
    def identity(self):return {'model_name':'fake-student','model_revision':'test-revision'}
    def save_checkpoint(self,path):
        Path(path).mkdir(parents=True,exist_ok=True)
        return str(path)
    def restore(self,path):self.restores.append(str(path))
    def infer(self,payload):
        self.inferences.append(payload)
        return {'raw':'```json\n{"document_type":"hospital_bill","evidence":["Synthetic hospital invoice"],"requires_review":false}\n```',
                'truncated':False}
    def train(self,pairs,schemas,output_dir):
        self.trains.append(pairs)
        if self.crash_train:
            self.crash_train=False
            raise RuntimeError('simulated training interruption')
        return self.save_checkpoint(output_dir/'adapter')


class FakeEvaluator:
    def __init__(self,scores,fail_at=None):self.scores=scores;self.calls=0;self.fail_at=fail_at
    def evaluate(self,development,output_dir):
        self.calls+=1
        if self.calls==self.fail_at:raise RuntimeError('simulated evaluation interruption')
        return report(self.scores[min(self.calls-1,len(self.scores)-1)])


class FakeCurriculum:
    def __init__(self,base_config,development,output_dir):
        self.base_weights={t:1 for t in SCHEMAS}
        self.path=output_dir/'fake_records.jsonl'
        self.rows=read_jsonl(self.path)
        self.weights=[];self.closed=False
    def open(self):pass
    def close(self):self.closed=True
    def generate(self,target,weights):
        self.weights.append(weights)
        while len(self.rows)<target:self.rows.append(record(i=len(self.rows)))
        self.path.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        return self.rows
    def verified_pairs(self, target, weights):
        self.generate(target, weights)
        return self.answer_pairs()[:target]
    def answer_pairs(self):
        return [(r,{'document_type':'hospital_bill','evidence':['Synthetic hospital invoice'],'requires_review':False}) for r in self.rows]


class LoopCoreTests(unittest.TestCase):
    def test_cleaning_does_not_repair_json(self):
        self.assertEqual(clean_response('<think>private thoughts</think>\n```json\n{"x":1}\n```'),{'x':1})
        for bad in ('{"x":','Prefix {"x":1}','<think>unfinished','[]','{"x":1} trailing','{"x":1,"x":2}'):
            with self.assertRaises(ResponseError):clean_response(bad)
        with self.assertRaises(ResponseError):clean_response('{"x":1}',truncated=True)

    def test_completion_tokens_only(self):
        example=tokenize_teacher_pair(FakeTokenizer(),{'instruction':'x'},{'answer':'parent'},10)
        self.assertEqual(example['input_ids'],[1,2,3,4,5,9])
        self.assertEqual(example['completion_mask'],[0,0,0,1,1,1])
        with self.assertRaises(ValueError):tokenize_teacher_pair(FakeTokenizer(),{}, {},5)

    def test_task_aware_classification_and_invalid_evidence(self):
        source=record();gold={'document_type':'hospital_bill','evidence':['Synthetic hospital invoice'],'requires_review':False}
        good=score_response(source,gold,json.dumps(gold),SCHEMAS)
        self.assertEqual(good,{'task_score':1.0,'schema_evidence_valid':1.0})
        bad=dict(gold,document_type='claim_form')
        self.assertEqual(score_response(source,gold,json.dumps(bad),SCHEMAS)['task_score'],0)
        bad=dict(gold,evidence=['fabricated quote'])
        self.assertEqual(score_response(source,gold,json.dumps(bad),SCHEMAS)['schema_evidence_valid'],0)

    def test_complete_coverage_and_nan_guard(self):
        rows=[{'task':t,'task_score':1.0,'schema_evidence_valid':1.0} for t in SCHEMAS for _ in range(3)]
        summary=aggregate_scores(rows,SCHEMAS,3)
        self.assertEqual(summary['macro_task_score'],1)
        with self.assertRaises(ValueError):aggregate_scores(rows[:-1],SCHEMAS,3)
        rows[0]['task_score']=float('nan')
        with self.assertRaises(ValueError):aggregate_scores(rows,SCHEMAS,3)

    def test_stop_gate_requires_every_task_and_validity(self):
        cfg=load_loop_config()
        self.assertTrue(satisfactory(report(.95),cfg['thresholds'],SCHEMAS))
        self.assertFalse(satisfactory(report(.95,.5),cfg['thresholds'],SCHEMAS))
        bad=report(.95);bad['per_task']['clinical_coding']=.4
        self.assertFalse(satisfactory(bad,cfg['thresholds'],SCHEMAS))
        bad=report(.95);bad['per_task'].pop('clinical_coding')
        self.assertFalse(satisfactory(bad,cfg['thresholds'],SCHEMAS))
        self.assertFalse(satisfactory(report(float('nan')),cfg['thresholds'],SCHEMAS))

    def test_feedback_uses_scores_only(self):
        summary=report(.95);summary['per_task']['clinical_coding']=.2
        weights=feedback_weights({'claim_scrubbing':3},summary,SCHEMAS,.8,4)
        self.assertGreater(weights['clinical_coding'],weights['policy_summarization'])
        self.assertEqual(weights['claim_scrubbing'],3)

    def test_invalid_max_rounds_rejected(self):
        cfg=load_loop_config();cfg['max_rounds']=0
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json';atomic_json(path,cfg)
            with self.assertRaises(ValueError):load_loop_config(str(path))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.config=load_loop_config();self.config.update(output_dir=self.tmp.name,scenarios_per_round=2,max_rounds=3)
        self.base=load_config()
        self.development=[{'record':record(i=999),'reference':{}}]
        self.data_objects=[]
    def run_loop(self,runtime,evaluator):
        def data(*args):
            obj=FakeCurriculum(*args);self.data_objects.append(obj);return obj
        with patch('distillation_loop.controller.LightevalEvaluator', return_value=evaluator), \
             patch('distillation_loop.controller.frozen_development', return_value=self.development), \
             patch('distillation_loop.controller.CurriculumData', side_effect=data):
            return run_loop(runtime, self.base, self.config)

    def test_threshold_stop_and_completed_resume_no_training(self):
        runtime=FakeRuntime();evaluator=FakeEvaluator([.1,.7,.95])
        result=self.run_loop(runtime,evaluator)
        self.assertEqual(result['status'],'thresholds_met')
        self.assertEqual(result['completed_rounds'],2)
        self.assertEqual(len(runtime.trains),2)
        self.assertEqual([len(pairs) for pairs in runtime.trains],[2,4])
        result=self.run_loop(runtime,evaluator)
        self.assertEqual(len(runtime.trains),2)
        self.assertEqual(evaluator.calls,3)
        self.assertEqual(runtime.restores[-1],result['best_checkpoint'])
        for payload in runtime.inferences:self.assertEqual(set(payload),{'instruction','scenario','response_schema'})

    def test_max_round_limit_and_best_checkpoint(self):
        self.config['max_rounds']=2
        runtime=FakeRuntime();evaluator=FakeEvaluator([.5,.6,.4])
        result=self.run_loop(runtime,evaluator)
        self.assertEqual(result['status'],'max_rounds_reached')
        self.assertEqual(len(runtime.trains),2)
        self.assertIn('round_01',result['best_checkpoint'])
        self.assertEqual(runtime.restores[-1],result['best_checkpoint'])

    def test_baseline_already_satisfies(self):
        runtime=FakeRuntime();result=self.run_loop(runtime,FakeEvaluator([.99]))
        self.assertEqual(result['completed_rounds'],0)
        self.assertEqual(result['status'],'thresholds_met')
        self.assertEqual(runtime.trains,[])
        self.assertEqual(self.data_objects,[])

    def test_unfinished_round_restarts_after_evaluation_failure(self):
        self.config['max_rounds']=1
        runtime=FakeRuntime();evaluator=FakeEvaluator([.1,.7,.95],fail_at=2)
        with self.assertRaisesRegex(RuntimeError,'evaluation interruption'):self.run_loop(runtime,evaluator)
        self.assertEqual(len(runtime.trains),1)
        result=self.run_loop(runtime,evaluator)
        self.assertEqual(result['completed_rounds'],1)
        self.assertEqual(len(runtime.trains),2)
        self.assertEqual(runtime.restores[-1],result['best_checkpoint'])

    def test_interrupted_training_restarts_from_previous_adapter(self):
        self.config['max_rounds']=1
        runtime=FakeRuntime(crash_train=True);evaluator=FakeEvaluator([.1,.95])
        with self.assertRaisesRegex(RuntimeError,'training interruption'):self.run_loop(runtime,evaluator)
        result=self.run_loop(runtime,evaluator)
        self.assertEqual(len(runtime.trains),2)
        self.assertIn('initial_adapter',runtime.restores[-2])
        self.assertEqual(result['completed_rounds'],1)
        self.assertTrue(all(obj.closed for obj in self.data_objects))

    def test_changed_configuration_and_development_rejected(self):
        runtime=FakeRuntime();self.run_loop(runtime,FakeEvaluator([.99]))
        self.config['max_rounds']=4
        with self.assertRaises(ValueError):self.run_loop(runtime,FakeEvaluator([.99]))
        self.config['max_rounds']=3
        self.development[0]['record']['instruction']='Changed frozen prompt'
        with self.assertRaises(ValueError):self.run_loop(runtime,FakeEvaluator([.99]))

    def test_no_replay_uses_new_pairs_only(self):
        self.config['max_rounds']=2;self.config['training']['replay_previous_rounds']=False
        runtime=FakeRuntime();self.run_loop(runtime,FakeEvaluator([.1,.4,.7]))
        self.assertEqual([len(pairs) for pairs in runtime.trains],[2,2])

    def test_diagnostics_are_not_training_targets(self):
        self.config['max_rounds']=1
        runtime=FakeRuntime();self.run_loop(runtime,FakeEvaluator([.1,.95]))
        rows=read_jsonl(Path(self.tmp.name)/'round_01/student_responses.jsonl')
        self.assertEqual(len(rows),2)
        self.assertTrue(all(r['cleaned'] is not None for r in rows))
        self.assertTrue(all('purpose' in r for r in rows))
        for r,answer in runtime.trains[0]:
            self.assertNotIn('raw',answer)
            self.assertNotIn('purpose',answer)

    def test_filters_invalid_training_pairs_and_keeps_good_subset(self):
        self.config['max_rounds'] = 1
        original = FakeCurriculum.answer_pairs
        with patch.object(FakeCurriculum, 'answer_pairs', lambda data: original(data)[:1]):
            runtime = FakeRuntime()
            result = self.run_loop(runtime, FakeEvaluator([.1, .95]))
        self.assertEqual(result['completed_rounds'], 1)
        self.assertEqual(len(runtime.trains[0]), 1)
        summary = json.loads((Path(self.tmp.name)/'round_01/training_filter.json').read_text())
        self.assertFalse(summary['target_reached'])
        self.assertEqual(summary['validated'], 1)

    def test_empty_training_subset_never_reaches_trainer(self):
        with patch.object(FakeCurriculum, 'answer_pairs', lambda data: []):
            runtime = FakeRuntime()
            with self.assertRaisesRegex(ValueError, 'Too few validated'):
                self.run_loop(runtime, FakeEvaluator([.1]))
        self.assertEqual(runtime.trains, [])

    def test_bad_baseline_does_not_start_training(self):
        class BadEvaluator:
            def evaluate(self,*args):return report(float('nan'))
        runtime=FakeRuntime()
        with self.assertRaises(ValueError):self.run_loop(runtime,BadEvaluator())
        self.assertEqual(runtime.trains,[])


if __name__=='__main__':unittest.main()
