import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from self_instruct_pipeline.config import load_config
from self_instruct_pipeline.parent_backend import OpenAIParent, ParentError, create_parent

MESSAGES = [{'role':'system','content':'Return the requested JSON.'}, {'role':'user','content':'Generate a scenario.'}]


def response(value):
    return io.BytesIO(json.dumps(value).encode())


def completed(value):
    return {'status':'completed','output':[
        {'type':'reasoning','summary':[]},
        {'type':'message','content':[{'type':'output_text','text':json.dumps({'result':value})}]}],
        'usage':{'input_tokens':20,'output_tokens':10,'total_tokens':30}}


class OpenAIBackendTests(unittest.TestCase):
    def test_default_provider_and_key_name(self):
        settings=load_config()['parent']
        self.assertEqual(settings['provider'],'openai')
        self.assertEqual(settings['model'],'gpt-6-luna')
        self.assertEqual(settings['api_key_env'],'OPENAI_API_KEY')
        self.assertIsInstance(create_parent(settings,api_key='TEST'),OpenAIParent)

    def test_responses_request_auth_and_usage(self):
        logs=[]
        def open_request(request,timeout):
            self.assertEqual(request.full_url,'https://api.openai.com/v1/responses')
            self.assertEqual(request.get_header('Authorization'),'Bearer TEST_SECRET')
            body=json.loads(request.data)
            self.assertEqual(body['model'],'gpt-6-luna')
            self.assertEqual(body['text']['format'],{'type':'json_object'})
            self.assertEqual(body['reasoning'],{'effort':'medium'})
            self.assertNotIn('temperature',body)
            self.assertNotIn('TEST_SECRET',json.dumps(body))
            self.assertFalse(body['store'])
            return response(completed({'ok':True}))
        client=OpenAIParent(load_config()['parent'],api_key='TEST_SECRET',opener=open_request,logger=logs.append)
        self.assertEqual(client.generate_json(MESSAGES,{'type':'object'}),{'ok':True})
        self.assertEqual(logs[0]['usage']['total_tokens'],30)
        self.assertNotIn('TEST_SECRET',json.dumps(logs))

    def test_json_mode_input_intent_when_task_does_not_mention_json(self):
        messages = [{'role': 'system', 'content': 'Produce JSON.'},
                    {'role': 'user', 'content': 'Explain this supplied policy.'}]
        def opener(request, timeout):
            body = json.loads(request.data)
            text = body['input'][0]['content']
            self.assertIn('json', text.lower())
            self.assertTrue(text.endswith(messages[1]['content']))
            self.assertEqual(messages[1]['content'], 'Explain this supplied policy.')
            return response(completed({}))
        client = OpenAIParent(load_config()['parent'], api_key='TEST', opener=opener)
        self.assertEqual(client.generate_json(messages, {'type':'object'}), {})

    def test_array_and_free_context_preserved(self):
        value=[{'scenario':{'context':{'claim_id':'C-1','requested_fields':['policy_number']}}}]
        client=OpenAIParent(load_config()['parent'],api_key='TEST',opener=lambda request,timeout:response(completed(value)))
        self.assertEqual(client.generate_json(MESSAGES,{'type':'array','items':{'type':'object'}}),value)

    def test_scenario_members_reach_pipeline_for_individual_validation(self):
        value = [{'valid': True}, {'malformed': True}]
        schema = {'type': 'array', 'items': {'type': 'object', 'required': ['valid']}}
        for operation in ('scenario_generation', 'scenario_audit'):
            client = OpenAIParent(load_config()['parent'], api_key='TEST',
                                  opener=lambda *a, **k: response(completed(value)))
            self.assertEqual(client.generate_json(MESSAGES, schema, operation), value)
        client = OpenAIParent(load_config()['parent'], api_key='TEST',
                              opener=lambda *a, **k: response(completed(value)))
        with self.assertRaises(ParentError): client.generate_json(MESSAGES, schema, 'teacher_answer')

    def test_model_access_checked(self):
        def open_request(request,timeout):
            self.assertEqual(request.full_url,'https://api.openai.com/v1/models/gpt-6-luna')
            return response({'id':'gpt-6-luna'})
        client=OpenAIParent(load_config()['parent'],api_key='TEST',opener=open_request)
        self.assertEqual(client.check_model(),{'model':'gpt-6-luna','provider':'openai'})

    def test_incomplete_refusal_invalid_json_and_schema(self):
        malformed=[
            {'status':'incomplete','incomplete_details':{'reason':'max_output_tokens'},'output':[]},
            {'status':'completed','output':[{'type':'message','content':[{'type':'refusal','refusal':'No'}]}]},
            {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'broken'}]}]},
            completed('wrong type'),
        ]
        for result in malformed:
            with self.subTest(result=result):
                client=OpenAIParent(load_config()['parent'],api_key='TEST',opener=lambda request,timeout:response(result))
                with self.assertRaises(ParentError):client.generate_json(MESSAGES,{'type':'object'})

    def test_disabled_reasoning_allows_temperature(self):
        settings=load_config()['parent'];settings['reasoning_effort']='none'
        def open_request(request,timeout):
            self.assertEqual(json.loads(request.data)['temperature'],0.0)
            return response(completed({}))
        client=OpenAIParent(settings,api_key='TEST',opener=open_request)
        client.generate_json(MESSAGES,{'type':'object'},temperature=0.0)

    def test_environment_auth_and_missing_key(self):
        with patch.dict(os.environ,{'OPENAI_API_KEY':'ENV_TEST'},clear=True):
            client=create_parent(load_config()['parent'])
            self.assertEqual(client.auth_headers()['Authorization'],'Bearer ENV_TEST')
        with patch.dict(os.environ,{},clear=True),self.assertRaisesRegex(ParentError,'OPENAI_API_KEY'):
            create_parent(load_config()['parent'])

    def test_bad_request_diagnostics_do_not_expose_body_or_key(self):
        logs, calls = [], []
        def rejected(request, timeout):
            calls.append(1)
            body = {'error': {'type': 'invalid_request_error', 'code': 'unsupported_parameter',
                              'param': 'service_tier', 'message': 'PRIVATE CLAIM and TEST_SECRET'}}
            raise urllib.error.HTTPError(request.full_url, 400, 'Bad Request', {},
                                         io.BytesIO(json.dumps(body).encode()))
        client = OpenAIParent(load_config()['parent'], api_key='TEST_SECRET', opener=rejected,
                              logger=logs.append)
        with self.assertRaisesRegex(ParentError, 'error_param=service_tier') as caught:
            client.generate_json(MESSAGES, {'type':'object'})
        self.assertEqual(len(calls), 1)
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(logs[0]['error_code'], 'unsupported_parameter')
        self.assertNotIn('PRIVATE CLAIM', str(caught.exception) + json.dumps(logs))
        self.assertNotIn('TEST_SECRET', str(caught.exception) + json.dumps(logs))

    def test_openai_retry_and_auth_failures(self):
        attempts=[]
        def limited(request,timeout):
            attempts.append(1)
            raise urllib.error.HTTPError(request.full_url,429,'not logged',{'Retry-After':'0'},None)
        settings=load_config()['parent'];settings['max_retries']=1
        client=OpenAIParent(settings,api_key='TEST',opener=limited,sleep=lambda _:None)
        with self.assertRaisesRegex(ParentError,'OpenAI HTTP 429'):client.check_model()
        self.assertEqual(len(attempts),2)
        attempts.clear()
        def unauthorized(request,timeout):
            attempts.append(1)
            raise urllib.error.HTTPError(request.full_url,401,'not logged',{},None)
        client=OpenAIParent(settings,api_key='TEST',opener=unauthorized)
        with self.assertRaisesRegex(ParentError,'OpenAI HTTP 401'):client.check_model()
        self.assertEqual(len(attempts),1)


if __name__=='__main__':unittest.main()
