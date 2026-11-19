"""Evidence relevance, targeted counterfactuals and canary preparation regression tests."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from autonomous_learning.scenarios import make_scenario, generate, expand, meaningful_variants
from autonomous_learning.rendering import render, verify, LAYOUT_COUNT, parse_text
from autonomous_learning.tasks import project, target_from_visible
from autonomous_learning.dataset import build, validate_example
from autonomous_learning.evaluation import score_case
from autonomous_learning.evidence import check_offsets, assess_evidence
from autonomous_learning.canary import api_plan, run_api
from autonomous_learning.storage import Storage
from autonomous_learning.model import MODEL_REVISION


def master(case='valid', condition=False, layout=0):
    return make_scenario(197, 0, 'hospitalization', condition, case, layout)


def case(task, s=None):
    s = s or master()
    return next(r for r in project(s, render(s)) if r['task'] == task)


class EvidenceTests(unittest.TestCase):
    def test_minimal_task_spans_and_exact_offsets(self):
        s = master('date_conflict')
        r = render(s)
        for row in project(s, r):
            validate_example(row)
            self.assertEqual(row['target'], target_from_visible(row))
            check_offsets(row['payload'], row['target'])
        classification = case('document_classification', s)
        self.assertEqual(len(classification['target']['evidence']), 3)
        self.assertTrue(all('Signature' not in e['quote'] and 'Patient' not in e['quote'] for e in classification['target']['evidence']))
        consistency = case('cross_document_consistency', s)
        self.assertEqual(len(consistency['target']['evidence']), 2)
        self.assertTrue(all('Service date' in e['quote'] for e in consistency['target']['evidence']))
        missing = case('missing_documents', expand(master(), 'remove_document'))
        self.assertEqual(missing['target']['evidence'], [])
        self.assertIn('inventory', missing['target']['source_references'])
        self.assertTrue(score_case(missing, {'raw': json.dumps(missing['target'])})['valid'])
        conditional = case('conditional_requirements')
        self.assertEqual(conditional['target']['evidence'], [])
        self.assertEqual(set(conditional['target']['source_references']), {'claim_facts/referral_required', 'fictional_rule_pack/conditional'})

    def test_offsets_alone_do_not_prove_relevance_or_sufficiency(self):
        row = case('document_classification')
        good = score_case(row, {'raw': json.dumps(row['target'])})
        self.assertEqual(good['score'], 1)
        irrelevant = deepcopy(row['target'])
        irrelevant['evidence'] = case('field_extraction')['target']['evidence'][:1]
        result = score_case(row, {'raw': json.dumps(irrelevant)})
        self.assertTrue(result['finding_correct'])
        self.assertTrue(result['evidence_offsets_valid'])
        self.assertFalse(result['evidence_relevant'])
        self.assertFalse(result['valid'])
        insufficient = deepcopy(row['target'])
        insufficient['evidence'] = insufficient['evidence'][:1]
        self.assertFalse(score_case(row, {'raw': json.dumps(insufficient)})['evidence_sufficient'])
        wrong = deepcopy(row['target'])
        wrong['result'][0]['document_type'] = 'prescription'
        result = score_case(row, {'raw': json.dumps(wrong)})
        self.assertTrue(result['evidence_offsets_valid'])
        self.assertFalse(result['finding_correct'])
        self.assertTrue(result['unsupported_claims'])

    def test_review_flags_malformed_outputs_and_unavailable_semantics(self):
        row = case('conditional_requirements', master(condition=None))
        wrong = deepcopy(row['target'])
        wrong['requires_review'] = False
        score = score_case(row, {'raw': json.dumps(wrong)})
        self.assertTrue(score['missed_required_review'])
        self.assertFalse(score['review_flag_correct'])
        self.assertFalse(score_case(row, {'raw': '{'})['schema_valid'])
        self.assertTrue(score_case(row, {'raw': '{}', 'truncated': True})['incomplete'])
        external = deepcopy(case('document_classification'))
        page = external['payload']['scenario']['documents'][0]
        page['text'] = 'Independently authored discharge narrative.'
        target = deepcopy(external['target'])
        target['evidence'] = [{'document_id': page['document_id'], 'page_number': 1, 'start': 0,
                               'end': len(page['text']), 'quote': page['text']}]
        external['target'] = target
        result = score_case(external, {'raw': json.dumps(target)})
        self.assertFalse(result['semantic_grounding_available'])
        self.assertEqual(result['grounding_basis'], 'semantic_grounding_unavailable')
        result = score_case(external, {'raw': json.dumps(target)}, reference_reviewed=True)
        self.assertTrue(result['semantic_grounding_available'])
        self.assertEqual(result['grounding_basis'], 'independently_reviewed_reference_anchors')

    def test_issue_explanations_do_not_cite_unneeded_complete_fields(self):
        s = master('unsigned')
        row = case('finding_explanation', s)
        affected = s['documents'][-1]['id']
        quotes = [e['quote'] for e in row['target']['evidence'] if e['document_id'] == affected]
        self.assertNotIn('Patient identifier: ' + s['facts']['patient_id'], quotes)
        self.assertIn('Signature: absent', quotes)

    def test_distractor_and_overinclusive_evidence_rejected(self):
        row = case('field_extraction', master(layout=5))
        target = deepcopy(row['target'])
        page = row['payload']['scenario']['documents'][0]
        quote = 'Processing desk: SYNTHETIC ADMIN ONLY'
        start = page['text'].index(quote)
        target['evidence'].append({'document_id': page['document_id'], 'page_number': 1,
                                   'start': start, 'end': start + len(quote), 'quote': quote})
        self.assertFalse(score_case(row, {'raw': json.dumps(target)})['evidence_relevant'])
        row['target'] = target
        with self.assertRaises(ValueError):
            validate_example(row)

    def test_out_of_bounds_offsets_and_paths_rejected(self):
        row = case('document_classification')
        bad = deepcopy(row['target'])
        page = row['payload']['scenario']['documents'][0]
        bad['evidence'] = [{'document_id': page['document_id'], 'page_number': 1,
                            'start': 0, 'end': len(page['text']) + 100, 'quote': page['text']}]
        with self.assertRaises(ValueError):
            check_offsets(row['payload'], bad)
        bad = deepcopy(row['target'])
        bad['source_references'] = ['documents/-1/text']
        with self.assertRaises(ValueError):
            check_offsets(row['payload'], bad)

    def test_review_proofs_include_unreadable_sources_and_inventory_is_task_scoped(self):
        source = make_scenario(123, 0, 'diagnostic', False, 'unreadable', 0)
        for task in ('field_extraction', 'cross_document_consistency'):
            row = case(task, source)
            self.assertTrue(row['target']['requires_review'])
            self.assertTrue(any(e['quote'] == 'Contents: unreadable' for e in row['target']['evidence']))
            validate_example(row)
        classification = case('document_classification', source)
        self.assertTrue(all(item['document_type'] is None for item in classification['payload']['scenario']['inventory'] if item['present']))
        missing = case('missing_documents', source)
        self.assertTrue(all(item['document_type'] is not None for item in missing['payload']['scenario']['inventory']))
        self.assertEqual(missing['target']['evidence'], [])
        legacy = deepcopy(missing)
        for item in legacy['payload']['scenario']['inventory']:
            if item['present']:
                item['document_type'] = None
        target = target_from_visible(legacy)
        self.assertEqual(len(target['evidence']), len(legacy['payload']['scenario']['documents']))
        classification['payload']['scenario']['inventory'][0]['document_type'] = 'claim_form'
        with self.assertRaises(ValueError):
            validate_example(classification)

    def test_unreviewed_semantics_never_become_a_numeric_transfer_score(self):
        from autonomous_learning.evaluation import evaluate, compare
        from autonomous_learning.storage import digest
        from distillation_loop.adapter_identity import adapter_files
        from autonomous_learning import CONTRACT_VERSION
        from autonomous_learning.planner import planner_input
        row = deepcopy(case('document_classification'))
        page = row['payload']['scenario']['documents'][0]
        page['text'] = 'An independently authored report without deterministic grammar.'
        row['target']['evidence'] = [{'document_id': page['document_id'], 'page_number': 1,
                                     'start': 0, 'end': len(page['text']), 'quote': page['text']}]
        benchmark = {'version': 1, 'tier': 'probes', 'process': 'independent', 'independently_reviewed': False, 'cases': [row]}
        with tempfile.TemporaryDirectory() as d:
            adapter = Path(d)
            (adapter / 'adapter_config.json').write_text('{}')
            (adapter / 'adapter_model.safetensors').write_bytes(b'fake fixture')
            generations = {'benchmark_id': digest(benchmark), 'adapter_sha256': digest(adapter_files(adapter)),
                           'contract_version': CONTRACT_VERSION, 'model_name': 'Qwen/Qwen3-8B',
                           'predictions': {row['id']: {'raw': json.dumps(row['target']), 'truncated': False}}}
            report = evaluate(benchmark, generations, adapter)
            self.assertFalse(report['grounded_score_available'])
            self.assertIsNone(report['macro_task_score'])
            self.assertIsNone(report['out_of_generator_transfer_accuracy'])
            self.assertFalse(compare(report, report)['accepted'])
            self.assertNotIn('per_task', planner_input(report)['performance'])


class ScenarioQualityTests(unittest.TestCase):
    def test_fully_valid_controls_exclude_unknown_conditions(self):
        unknown = master(condition=None)
        self.assertEqual(unknown['provenance']['case'], 'unknown_applicability')
        self.assertTrue(any(f['requires_review'] for f in unknown['findings']))
        for seed in range(10):
            for s in generate(8, seed)['scenarios']:
                if s['provenance']['case'] == 'valid':
                    self.assertIsNotNone(s['facts']['referral_required'])
                    self.assertFalse(any(f['requires_review'] for f in s['findings']))

    def test_explicit_target_not_inventory_last_and_no_unrelated_mutations(self):
        s = master(condition=True)
        discharge = next(d for d in s['documents'] if d['type'] == 'discharge_summary')
        referral = next(d for d in s['documents'] if d['type'] == 'referral')
        variant = expand(s, 'remove_document', discharge['id'])
        self.assertTrue(next(d for d in variant['documents'] if d['id'] == referral['id'])['present'])
        for original, changed in zip(s['documents'], variant['documents']):
            if original['id'] != discharge['id']:
                self.assertEqual(original, changed)
        self.assertEqual(variant['family_id'], s['family_id'])
        with self.assertRaises(ValueError):
            expand(s, 'remove_document', 'MISSING-ID')
        mismatch = expand(s, 'introduce_date_mismatch', discharge['id'], s['documents'][0]['id'])
        dates = [f for f in mismatch['findings'] if f['rule_id'] == 'consistency.service_dates']
        self.assertEqual(dates[0]['status'], 'issue_detected')
        fixed = expand(mismatch, 'fix_dates', discharge['id'], s['documents'][0]['id'])
        self.assertEqual(fixed['documents'], s['documents'])
        with self.assertRaises(ValueError):
            expand(s, 'introduce_date_mismatch', discharge['id'], discharge['id'])
        optional = deepcopy(s)
        optional['facts']['referral_required'] = False
        from autonomous_learning.scenarios import seal
        seal(optional)
        with self.assertRaises(ValueError):
            expand(optional, 'remove_document', referral['id'])

    def test_noop_or_unrelated_state_repairs_not_silently_accepted(self):
        s = master('missing')
        with self.assertRaises(ValueError):
            expand(s, 'make_unreadable')
        with self.assertRaises(ValueError):
            expand(master(), 'restore_document')
        for s in generate(8, 81)['scenarios']:
            versions = meaningful_variants(s, 3)
            self.assertGreaterEqual(len(versions), 2)
            self.assertEqual(len({v['content_hash'] for v in versions}), len(versions))
            for v in versions:
                verify(v, render(v))

    def test_all_surface_styles_preserve_facts_and_labels(self):
        s = master()
        expected = s['findings']
        texts = set()
        for layout in range(LAYOUT_COUNT):
            candidate = deepcopy(s)
            candidate['layout'] = layout
            from autonomous_learning.scenarios import seal
            seal(candidate)
            r = render(candidate)
            verify(candidate, r)
            self.assertEqual(candidate['findings'], expected)
            texts.add(r['pages'][0]['text'])
            for row in project(candidate, r):
                validate_example(row)
        self.assertGreaterEqual(len(texts), 8)
        tampered = render(master(layout=4))
        tampered['pages'][0]['text'] += '\n| Service date | 2026-01-01 |'
        from autonomous_learning.storage import digest
        tampered['render_hash'] = digest(tampered['pages'])
        with self.assertRaises(ValueError):
            verify(master(layout=4), tampered)

    def test_context_only_changes_with_different_labels_are_not_duplicates(self):
        unknown = master(condition=None)
        valid = expand(unknown, 'make_valid')
        rows = [case('conditional_requirements', unknown), case('conditional_requirements', valid)]
        self.assertNotEqual(rows[0]['target']['result'], rows[1]['target']['result'])
        selected = build(rows)
        self.assertEqual(len(selected['rows']), 2)
        self.assertEqual(selected['statistics']['duplicate_rate'], 0)

    def test_coverage_selection_deterministic_and_outcome_diverse(self):
        rows = []
        for s in generate(8, 91)['scenarios']:
            for v in [s] + meaningful_variants(s, 3):
                rows.extend(project(v, render(v)))
        left, right = build(rows, selection_seed=99), build(list(reversed(rows)), selection_seed=99)
        self.assertEqual(left['rows'], right['rows'])
        self.assertEqual(set(left['statistics']['task_distribution']), set(__import__('autonomous_learning.tasks', fromlist=['TASKS']).TASKS))
        self.assertGreater(left['statistics']['distinct_variants_retained'], 8)
        self.assertGreaterEqual(len(left['statistics']['intervention_distribution']), 3)
        self.assertIn('issue_detected', left['statistics']['distinct_findings_retained'])
        self.assertIn('valid', left['statistics']['control_distribution'])
        self.assertTrue(all(n <= 8 for n in __import__('collections').Counter(r['family_id'] for r in left['rows']).values()))


class CanaryTests(unittest.TestCase):
    def test_preparation_is_bounded_and_not_authorized(self):
        plan = api_plan()
        self.assertEqual(plan['planned_unique_requests'], 3)
        self.assertEqual(plan['max_attempts'], 6)
        self.assertLess(plan['estimated_reservation_all_attempts_usd'], plan['max_api_usd'])
        self.assertEqual(plan['settings']['model'], 'gpt-6-luna')
        self.assertEqual(len(MODEL_REVISION), 40)
        with tempfile.TemporaryDirectory() as d:
            storage = Storage(d)
            storage.initialize()
            with self.assertRaises(ValueError):
                run_api(storage, plan)
        self.assertTrue(any(d['readability'] == 'unreadable' for s in plan['masters'] for d in s['documents']))

    def test_continued_adapter_metadata_checks_and_canary_step_budget(self):
        from autonomous_learning.jobs import adapter_compatibility, prepare
        from autonomous_learning.dataset import export
        from autonomous_learning.planner import heuristic
        cfg = {'base_model_name_or_path': 'Qwen/Qwen3-8B', 'r': 16, 'lora_alpha': 32, 'lora_dropout': .05,
               'task_type': 'CAUSAL_LM', 'peft_type': 'LORA', 'bias': 'none',
               'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}
        with tempfile.TemporaryDirectory() as d:
            storage = Storage(d)
            storage.initialize()
            adapter = Path(d) / 'candidate'
            adapter.mkdir()
            (adapter / 'adapter_config.json').write_text(json.dumps(cfg))
            (adapter / 'adapter_model.safetensors').write_bytes(b'fake fixture, not trained weights')
            training = {'lora_rank': 16, 'lora_alpha': 32, 'lora_dropout': .05}
            self.assertTrue(adapter_compatibility(adapter, training)['compatible'])
            exported = export(storage, build(project(master(), render(master()))))
            job = prepare(storage, exported, {**heuristic(count=4), 'adapter_strategy': 'continued', 'max_gpu_usd': 1},
                          adapter, training_overrides={'max_steps_per_round': 2}, inference_overrides={'max_new_tokens': 1280})
            self.assertEqual(job['job']['config']['training']['max_steps_per_round'], 2)
            self.assertEqual(job['job']['model_revision'], MODEL_REVISION)
            self.assertEqual(job['job']['config']['training']['max_sequence_length'], 4096)
            cfg['target_modules'] = ['q_proj']
            (adapter / 'adapter_config.json').write_text(json.dumps(cfg))
            self.assertFalse(adapter_compatibility(adapter, training)['compatible'])
            with self.assertRaises(ValueError):
                prepare(storage, exported, {**heuristic(count=4), 'adapter_strategy': 'continued'}, adapter)

    def test_ambiguous_network_call_is_not_automatically_charged_again(self):
        import urllib.error
        from autonomous_learning.teacher import RenderQueue
        from self_instruct_pipeline.parent_backend import OpenAIParent
        settings = api_plan()['settings']
        attempts = []
        def opener(*args, **kwargs):
            attempts.append(1)
            raise urllib.error.URLError('connection dropped after request was sent')
        with tempfile.TemporaryDirectory() as d:
            storage = Storage(d)
            storage.initialize()
            queue = RenderQueue(storage)
            key = queue.enqueue(master()['documents'][-1], settings)
            factory = lambda: OpenAIParent(settings, api_key='mock-key', opener=opener, sleep=lambda x: None)
            queue.run(settings, client_factory=factory, max_calls=6, max_cost_usd='.02', concurrency=1, request_keys=[key])
            self.assertEqual(len(attempts), 1)
            self.assertEqual(queue.result(key)['status'], 'uncertain')
            queue.run(settings, client_factory=factory, max_calls=6, max_cost_usd='.02', concurrency=1, request_keys=[key])
            self.assertEqual(len(attempts), 1)

    def test_mock_canary_checks_changed_text_and_projects_verified_examples(self):
        from unittest.mock import patch
        from autonomous_learning.rendering import render_document
        plan = api_plan()
        for novel in (False, True):
            keys = iter(['key0', 'key1', 'key2'])
            results = {}
            for n, request in enumerate(plan['requests']):
                text = render_document(request['document'], 1)['text']
                if novel:
                    text = text.replace('Patient ID:', 'Pt ID:')
                results['key' + str(n)] = {'status': 'done', 'result': {'text': text}}
            with tempfile.TemporaryDirectory() as d, patch('autonomous_learning.canary.RenderQueue') as cls:
                storage = Storage(d)
                storage.initialize()
                queue = cls.return_value
                queue.enqueue.side_effect = lambda *args: next(keys)
                queue.result.side_effect = results.__getitem__
                summary = run_api(storage, plan, authorized=True)
                self.assertEqual(summary['useful_diversity_passed'], novel)
                self.assertEqual(summary['statistics']['total_examples'], 28)
                queue.run.assert_called_once()

    def test_replay_migration_reconstructs_targets_and_preserves_sources(self):
        from autonomous_learning.dataset import migrate_verified_replay
        row = case('document_classification')
        legacy = deepcopy(row)
        legacy['target']['result'][0]['document_type'] = 'wrong legacy label'
        migrated = migrate_verified_replay([legacy])[0]
        self.assertEqual(migrated['target'], row['target'])
        self.assertNotEqual(migrated['id'], legacy['id'])
        self.assertEqual(legacy['target']['result'][0]['document_type'], 'wrong legacy label')
        validate_example(migrated)


class NativeQualityTests(unittest.TestCase):
    @unittest.skipUnless(__import__('importlib.util', fromlist=['find_spec']).find_spec('lighteval'), 'Optional pinned LightEval CPU environment')
    def test_native_metric_enforces_grounding(self):
        from autonomous_learning.evaluation import make_lighteval_metric
        from lighteval.tasks.requests import Doc
        from lighteval.models.model_output import ModelResponse
        row = case('missing_documents', master('missing'))
        doc = Doc(query=json.dumps(row['payload']), choices=[json.dumps(row['target'])], gold_index=0,
                  specific={'case': row, 'truncated': False})
        metric = make_lighteval_metric()
        result = metric.compute_sample(model_response=ModelResponse(text=[json.dumps(row['target'])]), doc=doc)
        self.assertEqual(result['task_score'], 1)
        self.assertEqual(result['schema_evidence_valid'], 1)

    @unittest.skipUnless(__import__('importlib.util', fromlist=['find_spec']).find_spec('peft'), 'Optional pinned PEFT CPU environment')
    def test_runtime_prepares_native_rank32_rslora_without_loading_qwen_weights(self):
        import math
        from types import SimpleNamespace
        from unittest.mock import patch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from distillation_loop.student import StudentRuntime
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8))
        cfg = {'random_seed': 42, 'training': {'lora_rank': 32, 'lora_alpha': 32, 'lora_dropout': 0., 'use_rslora': True}}
        runtime = StudentRuntime(model, SimpleNamespace(pad_token_id=0), cfg)
        with patch('distillation_loop.student.require_single_gpu'):
            runtime.prepare()
        modules = [m for m in runtime.model.modules() if hasattr(m, 'lora_A')]
        self.assertEqual(len(modules), 7)
        self.assertTrue(runtime.model.peft_config['default'].use_rslora)
        for module in modules:
            self.assertAlmostEqual(module.scaling['default'], 32 / math.sqrt(32))


if __name__ == '__main__':
    unittest.main()
