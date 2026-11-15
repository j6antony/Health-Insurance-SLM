import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from distillation_loop.correction import adaptive_pairs, assess_candidate
from distillation_loop.trusted_templates import template_pair
from self_instruct_pipeline.parent_backend import ParentError
from self_instruct_pipeline.persistence import digest
import test_distillation_loop as fixtures


class Teacher:
    schemas = fixtures.SCHEMAS

    def __init__(self, assessment=None):
        self.assessment = assessment or {
            'verdict': 'needs_correction', 'error_categories': ['unsupported_conclusion'],
            'findings': 'Recalculate the result using the policy.',
        }
        self.calls = []

    def client(self):
        return self

    def generate_json(self, messages, schema, operation, temperature):
        self.calls.append(operation)
        if operation != 'student_assessment':
            raise AssertionError(operation)
        if isinstance(self.assessment, ParentError):
            raise self.assessment
        return self.assessment


class CorrectionTests(unittest.TestCase):
    def test_verified_reference_is_used_without_a_conclusion_check(self):
        record, target = template_pair(0)
        record['answer_checks'] = []
        prediction = {'raw': json.dumps(dict(target, answer='Wrong.')), 'truncated': False}
        teacher = Teacher()
        with tempfile.TemporaryDirectory() as folder:
            result = assess_candidate(teacher, record, target, prediction, folder, 'baseline')
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['reference'], target)
        self.assertEqual(teacher.calls, ['student_assessment'])

    def test_defective_teacher_reference_is_quarantined(self):
        record, target = template_pair(0)
        prediction = {'raw': json.dumps(target), 'truncated': False}
        teacher = Teacher({'verdict': 'reference_defective', 'error_categories': [],
                           'findings': 'Reference is contradicted by the source.'})
        with tempfile.TemporaryDirectory() as folder:
            result = assess_candidate(teacher, record, target, prediction, folder, 'baseline')
        self.assertEqual(result['status'], 'quarantined')
        self.assertEqual(result['reason'], 'Reference is contradicted by the source.')

    def test_invalid_student_output_retains_reference_without_api_call(self):
        record, target = template_pair(0)
        for raw, truncated in (('{bad', False), ('{}', False), (json.dumps(target), True)):
            with self.subTest(raw=raw, truncated=truncated), tempfile.TemporaryDirectory() as folder:
                teacher = Teacher()
                result = assess_candidate(teacher, record, target, {'raw': raw, 'truncated': truncated},
                                          folder, 'baseline')
                self.assertEqual(result['status'], 'accepted')
                self.assertEqual(result['reference'], target)
                self.assertEqual(result['assessment']['error_categories'], ['response_format'])
                self.assertEqual(teacher.calls, [])

    def test_resume_reuses_legacy_assessment_and_discards_rewrite(self):
        record, target = template_pair(0)
        prediction = {'raw': json.dumps(dict(target, answer='Wrong.')), 'truncated': False}
        teacher = Teacher()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / f"{digest(record['id'])}.json"
            path.write_text(json.dumps({
                'input_hash': digest([record, target, prediction, 'baseline']),
                'status': 'quarantined', 'assessment': teacher.assessment,
                'corrected': prediction, 'reason': 'Conclusion change lacks a deterministic check',
            }))
            result = assess_candidate(teacher, record, target, prediction, folder, 'baseline')
            self.assertEqual(result, assess_candidate(teacher, record, target, prediction, folder, 'baseline'))
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['reference'], target)
        self.assertNotIn('corrected', result)
        self.assertEqual(teacher.calls, [])

    def test_inconsistent_assessment_retains_target_but_ambiguity_quarantines(self):
        record, target = template_pair(0)
        prediction = {'raw': json.dumps(target), 'truncated': False}
        for verdict, status, feedback in (
            ('needs_correction', 'accepted', 'assessment_unavailable'),
            ('unverifiable', 'quarantined', 'unverifiable'),
        ):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as folder:
                teacher = Teacher({'verdict': verdict, 'error_categories': [], 'findings': ''})
                result = assess_candidate(teacher, record, target, prediction, folder, 'baseline')
                self.assertEqual(result['status'], status)
                self.assertEqual(result['assessment']['verdict'], feedback)

    def test_invalid_reference_is_quarantined_without_a_teacher_call(self):
        record, target = template_pair(0)
        teacher = Teacher()
        with tempfile.TemporaryDirectory() as folder:
            result = assess_candidate(teacher, record, {}, {'raw': json.dumps(target), 'truncated': False},
                                      folder, 'baseline')
        self.assertEqual(result['status'], 'quarantined')
        self.assertEqual(teacher.calls, [])

    def test_service_error_propagates_and_can_be_retried(self):
        record, target = template_pair(0)
        prediction = {'raw': json.dumps(target), 'truncated': False}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ParentError):
                assess_candidate(Teacher(ParentError('Temporary API failure')), record, target,
                                 prediction, folder, 'baseline')
            result = assess_candidate(Teacher(), record, target, prediction, folder, 'baseline')
        self.assertEqual(result['status'], 'accepted')

    def test_controller_uses_candidate_errors_and_replays_verified_targets(self):
        records = [template_pair(index) for index in (0, 3, 6, 9)]

        class Runtime(fixtures.FakeRuntime):
            def infer(self, payload):
                self.inferences.append(payload)
                record, target = next((r, a) for r, a in records if r['scenario'] == payload['scenario'])
                return {'raw': json.dumps(dict(target, answer='Wrong.')), 'truncated': False}

        observed = []
        pipeline = Teacher()

        class Curriculum:
            base_weights = {task: 1 for task in fixtures.SCHEMAS}

            def __init__(self, *args):
                self.pipeline = pipeline

            def open(self):
                pass

            def close(self):
                pass

            def verified_pairs(self, count, weights):
                observed.append((copy.deepcopy(pipeline.training_candidate_feedback), weights))
                return records[:count]

        fixture = fixtures.ControllerTests()
        fixture.setUp()
        self.addCleanup(fixture.tmp.cleanup)
        fixture.config.update(student_conditioned_training=True, max_rounds=2)
        runtime = Runtime()
        evaluator = fixtures.FakeEvaluator([.1, .2, .95])
        with patch('distillation_loop.controller.CurriculumData', Curriculum), \
             patch('distillation_loop.controller.frozen_development', return_value=fixture.development), \
             patch('distillation_loop.controller.LightevalEvaluator', return_value=evaluator):
            from distillation_loop.controller import run_loop
            state = run_loop(runtime, fixture.base, fixture.config)
            run_loop(runtime, fixture.base, fixture.config)
        task = records[0][0]['task']
        self.assertEqual(observed[0][0], {})
        self.assertEqual(observed[1][0][task]['error_counts'], {'unsupported_conclusion': 2})
        self.assertGreater(observed[1][1][task], observed[0][1][task])
        self.assertEqual(len(runtime.inferences), 4)
        self.assertEqual(runtime.trains, [records[:2], records])
        self.assertEqual(state['completed_rounds'], 2)
        self.assertTrue((Path(fixture.tmp.name) / 'round_02' / 'candidate_feedback.json').exists())


class MixedCurriculumTests(unittest.TestCase):
    def test_exact_mix_replays_prior_pairs_and_keeps_feedback_out_of_general_generation(self):
        feedback = {'field_extraction': {'error_counts': {'incorrect_extraction': 2}}}
        pipeline = SimpleNamespace(schemas={}, training_candidate_feedback=feedback)
        calls = []
        screened = []

        def generate(target, weights):
            calls.append((target, weights, pipeline.training_candidate_feedback))
            return [({'id': str(i), 'task': 'field_extraction', 'answer_checks': ['check']},
                     {'source': 'reference'}) for i in range(target)]

        def predict(runtime, records, schemas, directory):
            screened.extend(record['id'] for record in records)
            return {record['id']: {} for record in records}

        def assess(pipeline, record, reference, prediction, directory, checkpoint):
            return {'scenario': record, 'reference': reference,
                    'status': 'accepted', 'assessment': {'verdict': 'correct'}}

        data = SimpleNamespace(pipeline=pipeline, base_weights={'general': 1}, verified_pairs=generate)
        config = {'scenarios_per_round': 100, 'adaptive_pairs_per_round': 40,
                  'quality': {'minimum_new_pair_fraction': 1}}
        prior = [({'id': 'old'}, {'source': 'prior'})]
        with tempfile.TemporaryDirectory() as folder, patch(
                'distillation_loop.correction.assess_candidate', side_effect=assess):
            pairs, diagnostics = adaptive_pairs(data, None, Path(folder), {'last_checkpoint': 'prior'},
                                                config, {'steered': 1}, prior, predict)
        self.assertEqual(calls, [(60, {'general': 1}, {}), (100, {'steered': 1}, feedback)])
        self.assertEqual(screened, [str(i) for i in range(60, 100)])
        self.assertEqual([answer['source'] for _, answer in pairs],
                         ['prior'] + ['reference'] * 100)
        self.assertEqual(diagnostics['training_mix'],
                         {'general': 60, 'adaptive': 40, 'used_prior_feedback': True})

    def test_ambiguous_reference_is_replaced_with_a_fresh_candidate(self):
        pipeline = SimpleNamespace(schemas={}, training_candidate_feedback={})

        def generate(target, weights):
            return [({'id': str(i), 'task': 'field_extraction'}, {'source': 'reference'})
                    for i in range(target)]

        def assess(pipeline, record, reference, prediction, directory, checkpoint):
            accepted = int(record['id']) != 60
            return {'scenario': record, 'reference': reference,
                    'status': 'accepted' if accepted else 'quarantined',
                    'assessment': {'verdict': 'correct' if accepted else 'unverifiable',
                                   'error_categories': []}}

        data = SimpleNamespace(pipeline=pipeline, base_weights={}, verified_pairs=generate)
        config = {'scenarios_per_round': 100, 'adaptive_pairs_per_round': 40,
                  'quality': {'minimum_new_pair_fraction': 1}}
        with tempfile.TemporaryDirectory() as folder, patch(
                'distillation_loop.correction.assess_candidate', side_effect=assess) as assessment:
            pairs, diagnostics = adaptive_pairs(
                data, None, Path(folder), {'last_checkpoint': 'prior'}, config, {}, [],
                lambda runtime, records, schemas, directory: {r['id']: {} for r in records})
        self.assertEqual(assessment.call_count, 80)
        self.assertEqual(len({record['id'] for record, _ in pairs}), 100)
        self.assertNotIn('60', {record['id'] for record, _ in pairs})
        self.assertEqual(diagnostics['training_mix']['adaptive'], 40)

    def test_stratified_screening_refuses_to_skip_a_task(self):
        tasks = [f'task_{i}' for i in range(18)]
        pipeline = SimpleNamespace(schemas=dict.fromkeys(tasks), training_candidate_feedback={})

        def generate(target, weights):
            return [({'id': str(i), 'task': tasks[i % 18] if i < 120 else tasks[i % 6]}, {})
                    for i in range(target)]

        def assess(pipeline, record, reference, prediction, directory, checkpoint):
            return {'scenario': record, 'reference': reference, 'status': 'accepted',
                    'assessment': {'verdict': 'correct'}}

        data = SimpleNamespace(pipeline=pipeline, base_weights={}, verified_pairs=generate)
        config = {'scenarios_per_round': 200, 'adaptive_pairs_per_round': 80,
                  'balanced_curriculum': True, 'stratified_screening': True}
        with tempfile.TemporaryDirectory() as folder, patch(
                'distillation_loop.correction.assess_candidate', side_effect=assess):
            with self.assertRaisesRegex(ValueError, 'Adaptive screening floor not met'):
                adaptive_pairs(data, None, Path(folder), {'last_checkpoint': 'prior'}, config, {}, [],
                               lambda runtime, records, schemas, directory: {r['id']: {} for r in records})
