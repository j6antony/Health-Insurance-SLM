import copy
import io
import json
import os
import random
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from self_instruct_pipeline.config import ROOT, load_config
from self_instruct_pipeline.coverage_tracker import coverage_state
from self_instruct_pipeline.dataset_builder import export_dataset
from self_instruct_pipeline.deduplicator import Deduplicator, fingerprint
from self_instruct_pipeline.parent_backend import GeminiParent, ParentError
from self_instruct_pipeline.persistence import Store, digest, read_jsonl
from self_instruct_pipeline.pipeline import Pipeline
from self_instruct_pipeline.prompt_builder import generation_messages, teacher_messages, verification_messages
from self_instruct_pipeline.sampler import BalancedSeedSampler, group_seed_tasks
from self_instruct_pipeline.scheduler import WeightedDeficitScheduler, apportioned_counts
from self_instruct_pipeline.validation import (AUDIT_SCHEMA, ValidationError, batch_schema, validate_answer,
                                               validate_batch, validate_schema, validate_audit)
from run_self_instruct import main

SEEDS = [json.loads(x) for x in (ROOT / 'data/self_instruct/seed_tasks.jsonl').read_text().splitlines()]
SCHEMAS = json.loads((ROOT / 'data/self_instruct/task_schemas.json').read_text())['tasks']
GOOD_AUDIT = {'grounded': True, 'calculations_checked': True, 'uncertainty_handled': True,
              'unsupported_actions': False, 'safe_for_synthetic_sft': True, 'reasons': []}


def scenario(task='claim_scrubbing', i=0, difficulty='medium'):
    return {'generation_id': f'local-{i}', 'task': task, 'difficulty': difficulty,
            'instruction': f'Apply the explicitly synthetic checklist for package {i}; return the requested task schema.',
            'scenario': {'synthetic': True, 'documents': [{'document_id': 'D1', 'page_number': 1,
                          'text': f'Synthetic checklist for package {i}: claim form required. Form is missing.'}], 'context': {}},
            'response_schema_key': task, 'skills': ['missing_information'], 'case_type': 'insufficient_evidence',
            'source_mode': 'synthetic_rules', 'reference_ids': []}


class MockParent:
    def __init__(self, invalid=False, unsafe=False):
        self.calls = []
        self.index = 0
        self.invalid = invalid
        self.unsafe = unsafe

    def check_model(self):
        return {'model': 'mock-gemini'}

    def generate_json(self, messages, schema, operation='generation', temperature=None):
        self.calls.append((operation, messages))
        if operation == 'scenario_generation':
            request = json.loads(messages[1]['content'])['request']
            difficulty = [k for k,v in request['difficulty'].items() for _ in range(v)]
            result = []
            for task,count in request['task_quotas'].items():
                for _ in range(count):
                    self.index += 1
                    result.append(scenario(task, self.index, difficulty.pop()))
            return [] if self.invalid else result
        if operation == 'scenario_audit':
            return [{'generation_id': r['generation_id'], 'usable': not self.unsafe,
                     'coverage_dimensions': ['insufficient_ambiguous_unreadable'], 'duplicate_of': '', 'reasons': []}
                    for r in json.loads(messages[1]['content'])]
        if operation == 'teacher_answer':
            source = json.loads(messages[1]['content'])['scenario']['documents'][0]['text']
            answer = {'answer': 'The required form is missing.', 'evidence': [{'document_id': 'D1', 'quote': source}],
                      'missing_information': ['Claim form'], 'requires_review': True}
            for key in schema['required']:
                if key == 'issues': answer[key] = ['Required claim form missing']
                if key == 'suggested_codes': answer[key] = []
                if key == 'draft': answer[key] = 'Please review after documentation is supplied.'
                if key == 'criteria_assessment': answer[key] = ['Required documentation missing']
            if 'fields' in schema['required']: return {'fields': []}
            if 'document_type' in schema['required']:
                return {'document_type': 'other', 'evidence': [source], 'requires_review': True}
            return answer
        if operation == 'answer_audit':
            return GOOD_AUDIT.copy()
        raise AssertionError(operation)


class SamplingTests(unittest.TestCase):
    def test_all_tasks_and_no_id_mapping(self):
        groups = group_seed_tasks([dict(s, id='not-a-number') for s in reversed(SEEDS)])
        self.assertEqual(len(groups), 18)
        self.assertEqual(len(groups['claim_scrubbing']), 12)
        with self.assertRaises(ValueError): group_seed_tasks([{'id': 'seed_001'}])

    def test_weighted_deficits_and_failed_requests(self):
        s = WeightedDeficitScheduler(['a','b'], {'a':3,'b':1})
        self.assertEqual(s.next_quotas(40, random.Random(7)), {'a':30,'b':10})
        self.assertEqual(sum(s.counts.values()), 0)
        s.update([{'task':'a'}] * 30)
        self.assertEqual(s.next_quotas(8, random.Random(7)), {'b':8})

    def test_sample_size_diversity_and_reproducibility(self):
        scheduler = WeightedDeficitScheduler(SCHEMAS)
        a = BalancedSeedSampler(SEEDS, random.Random(8))
        b = BalancedSeedSampler(SEEDS, random.Random(8))
        quotas = {'claim_scrubbing':2, 'field_extraction':2}
        result = a.sample(quotas, scheduler, [])
        self.assertEqual(result, b.sample(quotas,scheduler,[]))
        self.assertEqual(len(result), 8)
        self.assertEqual(len({r['task'] for r in result}), 8)
        self.assertTrue(set(quotas) <= {r['task'] for r in result})
        self.assertTrue(all('output' in r and 'input' in r for r in result))

    def test_machine_cap_and_invalid_pool(self):
        scheduler = WeightedDeficitScheduler(SCHEMAS)
        records = [{**scenario(i=i), 'id': f'm-{i}', 'validation_status':'scenario_checks_passed'} for i in range(10)]
        records += [{**scenario(i=99), 'id': 'REJECTED', 'validation_status':'rejected'}]
        sampler = BalancedSeedSampler(SEEDS, random.Random(1))
        examples = sampler.sample({'claim_scrubbing':8},scheduler,records)
        machine = [r for r in examples if r['id'].startswith('m-')]
        self.assertLessEqual(len(machine),2)
        self.assertTrue(all('output' not in r for r in machine))
        self.assertNotIn('REJECTED', {r['id'] for r in examples})


class ValidationTests(unittest.TestCase):
    def test_batch_contract_and_forbidden_outputs(self):
        row = scenario()
        validate_batch([row],{'claim_scrubbing':1},{'medium':1},SCHEMAS,[])
        for mutation in ('task','response_schema_key','output','page','documents','context'):
            bad = copy.deepcopy(row)
            if mutation == 'task': bad['task']='unknown-task'
            elif mutation == 'response_schema_key': bad['response_schema_key']='field_extraction'
            elif mutation == 'output': bad['output']='answer leakage'
            elif mutation == 'page': bad['scenario']['documents'][0]['page_number']=0
            elif mutation == 'documents': bad['scenario']['documents']=[]
            else: bad['scenario']['context']['expected_answer']='missing form'
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                validate_batch([bad],{'claim_scrubbing':1},{'medium':1},SCHEMAS,[])

    def test_count_quota_difficulty_and_duplicate_ids(self):
        row=scenario()
        for records,quotas,diff in [([],{'claim_scrubbing':1},{'medium':1}),
                                   ([row],{'field_extraction':1},{'medium':1}),
                                   ([row],{'claim_scrubbing':1},{'hard':1}),
                                   ([row,row],{'claim_scrubbing':2},{'medium':2})]:
            with self.assertRaises(ValidationError): validate_batch(records,quotas,diff,SCHEMAS,[])

    def test_evidence_schema_uncertainty_codes(self):
        row=scenario()
        source=row['scenario']['documents'][0]['text']
        good={'answer':'Missing form','evidence':[{'document_id':'D1','quote':source}],
              'missing_information':['form'],'requires_review':True,'issues':['missing form']}
        validate_answer(row,good,SCHEMAS[row['task']])
        for mutation in ('quote','id','review','extra'):
            bad=copy.deepcopy(good)
            if mutation=='quote': bad['evidence'][0]['quote']='invented quote'
            elif mutation=='id':bad['evidence'][0]['document_id']='WRONG'
            elif mutation=='review':bad['requires_review']=False
            else:bad['extra']='unexpected'
            with self.assertRaises(ValidationError):validate_answer(row,bad,SCHEMAS[row['task']])
        bad={k:v for k,v in good.items() if k!='issues'}
        bad['suggested_codes']=['INVENTED-CODE']
        with self.assertRaises(ValidationError):validate_answer(row,bad,SCHEMAS['clinical_coding'])

    def test_extraction_page_and_value(self):
        row=scenario('field_extraction')
        row['scenario']['documents'][0]['text']='Policy: SYN-1.'
        answer={'fields':[{'field_name':'policy_number','value':'SYN-1','evidence':'Policy: SYN-1.',
                           'page_number':1,'requires_review':False}]}
        validate_answer(row,answer,SCHEMAS['field_extraction'])
        answer['fields'][0]['page_number']=2
        with self.assertRaises(ValidationError):validate_answer(row,answer,SCHEMAS['field_extraction'])
        answer['fields'][0]['page_number']=1
        answer['fields'][0]['value']='SYN-9'
        with self.assertRaises(ValidationError):validate_answer(row,answer,SCHEMAS['field_extraction'])

    def test_audit_and_arithmetic_failure(self):
        audit=GOOD_AUDIT.copy();audit['calculations_checked']=False
        with self.assertRaises(ValidationError):validate_audit(audit)
        row=scenario('field_extraction')
        row['scenario']['context']['numeric_constraints']=[{'operands':[120,80],'operation':'sum','claimed_result':250}]
        with self.assertRaises(ValidationError):validate_answer(row,{'fields':[]},SCHEMAS['field_extraction'])

    def test_duplicates_and_reasoning_contrasts(self):
        row=scenario()
        dedup=Deduplicator([row])
        self.assertEqual(dedup.check(row),'exact_duplicate')
        distinct=copy.deepcopy(row)
        distinct['scenario']['documents'][0]['text']='Synthetic checklist: form required. Form present but unreadable.'
        self.assertIsNone(dedup.check(distinct))

    def test_no_hidden_metadata_teacher_or_auditor(self):
        row={**scenario(),'id':'hidden-id','seed_ids':['seed_001'],'secret_answer':'DO NOT PASS'}
        for messages in (teacher_messages(row,SCHEMAS),verification_messages(row,{},SCHEMAS)):
            payload=json.loads(messages[1]['content'])
            for field in ('case_type','difficulty','skills','seed_ids','secret_answer','id'):
                self.assertNotIn(field,payload)

    def test_approved_reference_integrity(self):
        row=scenario()
        row['source_mode']='approved_references'
        row['reference_ids']=['REF-1']
        references=[{'id':'REF-1','text':'Approved immutable excerpt.', 'version':'v7','jurisdiction':'Example Region'}]
        row['scenario']['documents'].append({'document_id':'D2','page_number':1,'text':references[0]['text']})
        row['scenario']['context']['source_metadata']={'version':'v7','jurisdiction':'Example Region'}
        validate_batch([row],{'claim_scrubbing':1},{'medium':1},SCHEMAS,references)
        row['scenario']['documents'][1]['text']='Modified excerpt.'
        with self.assertRaises(ValidationError):validate_batch([row],{'claim_scrubbing':1},{'medium':1},SCHEMAS,references)

    def test_overlapping_coverage(self):
        row={**scenario(),'coverage_dimensions':['conflicting_or_misleading','conditional_logic_or_date']}
        s=WeightedDeficitScheduler(SCHEMAS);s.update([row])
        state=coverage_state([row],s)
        self.assertEqual(state['observed_coverage_counts']['conflicting_or_misleading'],1)
        self.assertEqual(state['observed_coverage_counts']['conditional_logic_or_date'],1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config=load_config()
        self.config['paths']['output_dir']=self.tmp.name
        self.config['generation']['target_accepted_scenarios']=2
        self.config['generation']['batch_size']=2
        self.config['generation']['max_retries_per_batch']=0
        self.config['generation']['max_consecutive_failed_batches']=2

    def test_dry_run_never_calls_parent_and_prompt_contract(self):
        mock=MockParent()
        p=Pipeline(self.config,mock)
        plan=p.plan()
        self.assertEqual(mock.calls,[])
        self.assertNotIn('```',plan['messages'][0]['content'])
        payload=json.loads(plan['messages'][1]['content'])
        self.assertEqual(set(payload['schemas']),set(plan['task_quotas']))
        self.assertEqual(len(payload['seeds']),8)
        self.assertEqual(payload['request']['count'],2)
        self.assertEqual(sum(payload['request']['difficulty'].values()),2)

    def test_held_out_exclusions_never_enter_generation_prompts(self):
        held_out=scenario(i=999)
        held_out['instruction']='UNIQUE_HELD_OUT_SECRET'
        p=Pipeline(self.config,MockParent(),exclusions=[held_out])
        plan=p.plan()
        self.assertNotIn('UNIQUE_HELD_OUT_SECRET',json.dumps(plan['messages']))
        self.assertFalse(any(r.get('instruction')=='UNIQUE_HELD_OUT_SECRET' for r in p.dedup.rows))
        self.assertEqual(p.exclusion_dedup.check(held_out),'exact_duplicate')

    def test_full_mock_run_resume_export_and_no_leakage(self):
        mock=MockParent()
        p=Pipeline(self.config,mock);p.initialize_manifest()
        p.generate();p.answer();self.assertEqual(p.export(),2);p.checkpoint()
        calls=len(mock.calls)
        resumed=Pipeline(self.config,mock);resumed.initialize_manifest()
        self.assertEqual(sum(resumed.scheduler.counts.values()),2)
        resumed.generate();resumed.answer();self.assertEqual(resumed.export(),2)
        self.assertEqual(len(mock.calls),calls)
        self.assertEqual(len(p.store.read('validated_answers.jsonl')),2)
        rows=p.store.read('qwen3_8b_chat.jsonl')
        self.assertEqual(len(rows),2)
        for row in rows:
            self.assertEqual(set(row),{'messages'})
            self.assertEqual([m['role'] for m in row['messages']],['user','assistant'])
            user=json.loads(row['messages'][0]['content'])
            self.assertEqual(set(user),{'instruction','scenario','response_schema'})
        config=copy.deepcopy(self.config);config['parent']['model']='gemini-other'
        with self.assertRaises(ValueError):Pipeline(config,mock).initialize_manifest()

    def test_failures_bounded_and_rejected_not_counted(self):
        p=Pipeline(self.config,MockParent(invalid=True));p.initialize_manifest()
        with self.assertRaisesRegex(ValueError,'consecutive failed'):p.generate()
        self.assertEqual(p.accepted,[])
        self.assertEqual(sum(p.scheduler.counts.values()),0)
        self.assertEqual(len(p.store.read('rejected_records.jsonl')),2)

    def test_unsafe_scenario_never_accepted(self):
        p=Pipeline(self.config,MockParent(unsafe=True));p.initialize_manifest()
        with self.assertRaises(ValueError):p.generate()
        self.assertEqual(p.accepted,[])

    def test_duplicate_store_and_partial_line_fail(self):
        store=Store(self.tmp.name)
        store.append('accepted_scenarios.jsonl',{'id':'x'})
        store.append('accepted_scenarios.jsonl',{'id':'x'})
        with self.assertRaises(ValueError):store.unique_records('accepted_scenarios.jsonl','id')
        store.path('broken.jsonl').write_text('{"id":')
        with self.assertRaises(ValueError):store.read('broken.jsonl')

    def test_empty_answer_export_excludes_unverified(self):
        p=Pipeline(self.config,MockParent());p.initialize_manifest();p.generate()
        self.assertEqual(p.export(),0)

    def test_semantic_duplicate_is_quarantined(self):
        class DuplicateParent(MockParent):
            def generate_json(self,messages,schema,operation='generation',temperature=None):
                value=super().generate_json(messages,schema,operation,temperature)
                if operation=='scenario_audit':
                    visible=json.loads(messages[1]['content'])
                    for audit,item in zip(value,visible):
                        audit['duplicate_of']=item['related_examples'][0]['id']
                return value
        p=Pipeline(self.config,DuplicateParent());p.initialize_manifest()
        with self.assertRaises(ValueError):p.generate()
        self.assertEqual(len(p.accepted),0)
        self.assertTrue(all(r['reason'].startswith('semantic_duplicate:') for r in p.store.read('rejected_records.jsonl')))

    def test_bad_late_audit_cannot_partially_commit_batch(self):
        class BadAuditParent(MockParent):
            def generate_json(self,messages,schema,operation='generation',temperature=None):
                value=super().generate_json(messages,schema,operation,temperature)
                if operation=='scenario_audit':value[-1]['duplicate_of']='UNKNOWN-ID'
                return value
        p=Pipeline(self.config,BadAuditParent());p.initialize_manifest()
        with self.assertRaises(ValueError):p.generate()
        self.assertEqual(len(p.accepted),0)

    def test_invalid_teacher_answer_never_exported(self):
        class InvalidAnswerParent(MockParent):
            def generate_json(self,messages,schema,operation='generation',temperature=None):
                value=super().generate_json(messages,schema,operation,temperature)
                if operation=='teacher_answer':value['EXTRA']='unsupported'
                return value
        p=Pipeline(self.config,InvalidAnswerParent());p.initialize_manifest();p.generate()
        with self.assertRaises(ValueError):p.answer()
        self.assertEqual(p.export(),0)
        self.assertEqual(p.store.read('validated_answers.jsonl'),[])


class BackendTests(unittest.TestCase):
    def settings(self):
        return load_config()['parent']

    def response(self,data):
        return io.BytesIO(json.dumps(data).encode())

    def test_json_request_and_usage(self):
        events=[]
        def open_request(request,timeout):
            body=json.loads(request.data)
            self.assertEqual(body['generationConfig']['responseMimeType'],'application/json')
            self.assertEqual(body['systemInstruction']['parts'][0]['text'],'system')
            self.assertNotIn('TEST_SECRET',request.full_url)
            return self.response({'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{"ok": true}'}]}}],
                                  'usageMetadata':{'totalTokenCount':7}})
        client=GeminiParent(self.settings(),api_key='TEST_SECRET',opener=open_request,logger=events.append)
        self.assertEqual(client.generate_json([{'role':'system','content':'system'},{'role':'user','content':'user'}],{'type':'object'}),{'ok':True})
        self.assertNotIn('TEST_SECRET',json.dumps(events))
        self.assertEqual(events[0]['usage']['totalTokenCount'],7)

    def test_transient_retries_and_auth_no_retry(self):
        tries=[]
        def fail(request,timeout):
            tries.append(1)
            raise urllib.error.HTTPError(request.full_url,429,'limited',{'Retry-After':'0'},None)
        settings=self.settings();settings['max_retries']=2
        client=GeminiParent(settings,api_key='secret',opener=fail,sleep=lambda _:None)
        with self.assertRaises(ParentError):client.check_model()
        self.assertEqual(len(tries),3)
        tries.clear()
        def auth(request,timeout):
            tries.append(1)
            raise urllib.error.HTTPError(request.full_url,401,'secret',{},None)
        client=GeminiParent(settings,api_key='secret',opener=auth)
        with self.assertRaisesRegex(ParentError,'HTTP 401'):client.check_model()
        self.assertEqual(len(tries),1)

    def test_truncated_and_non_json(self):
        for finish,text in [('MAX_TOKENS','[]'),('STOP','not json')]:
            client=GeminiParent(self.settings(),api_key='secret',opener=lambda request,timeout:self.response({
                'candidates':[{'finishReason':finish,'content':{'parts':[{'text':text}]}}]}))
            with self.assertRaises(ParentError):client.generate_json([{'role':'system','content':'s'},{'role':'user','content':'u'}],{})

    def test_missing_env_and_invalid_config(self):
        with patch.dict(os.environ,{},clear=True),self.assertRaises(ParentError):GeminiParent(self.settings())
        config=load_config();self.assertEqual(config['parent']['model'],'gemini-3.5-flash')
        with tempfile.TemporaryDirectory() as d:
            config['sampling']['mode']='random'
            path=Path(d)/'bad.json';path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):load_config(path)


if __name__=='__main__':
    unittest.main()
