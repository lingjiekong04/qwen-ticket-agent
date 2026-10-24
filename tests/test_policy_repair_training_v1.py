"""CPU/mock tests only: no CUDA initialization, model load or real training."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
from extensions.ticket_policy_repair_training_v1 import support as s, train as t, evaluate as e


class Tensor:
    def __init__(self, values): self.values, self.shape = list(values), (len(values),)
    def detach(self): return self
    def cpu(self): return self
    def to(self, dtype=None): return self
    @property
    def dtype(self): return 'fake-float'
    def numel(self): return len(self.values)
    def all(self): return NS(item=lambda: all(self.values))


FAKE_TORCH = ModuleType('torch')
FAKE_TORCH.equal = lambda a, b: a.values == b.values
FAKE_TORCH.isfinite = lambda a: Tensor([math.isfinite(v) for v in a.values])


class ConfigAndInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes, cls.records, cls.indices, cls.longest = s.inputs()

    def reject_config(self, change, stage='train'):
        config = s.config_for(stage)
        change(config)
        with self.assertRaises(ValueError): s.check_config(config, stage)

    def test_configs_exact(self):
        for stage in ('preflight', 'train'): s.check_config(s.read(s.config_path(stage)), stage)

    def test_actual_step_arithmetic(self):
        c = s.config_for('train')
        self.assertEqual(1200 // c['gradient_accumulation_steps'] * c['num_train_epochs'], 300)
        self.assertNotIn('max_steps', c)

    def test_wrong_initializer(self): self.reject_config(lambda c: c.update(adapter_name_or_path=s.old.OLD))
    def test_new_lora(self): self.reject_config(lambda c: c.update(create_new_adapter=True))
    def test_resume(self): self.reject_config(lambda c: c.update(resume_from_checkpoint='checkpoint-300'))
    def test_old_data_mix(self): self.reject_config(lambda c: c.update(dataset=c['dataset'] + ',ticket_sft_v3_train'))
    def test_dev_in_train(self): self.reject_config(lambda c: c.update(dataset='ticket_agent_v2_dev'))
    def test_train_on_prompt(self): self.reject_config(lambda c: c.update(train_on_prompt=True))
    def test_mask_history(self): self.reject_config(lambda c: c.update(mask_history=True))
    def test_packing(self): self.reject_config(lambda c: c.update(packing=True))
    def test_cutoff(self): self.reject_config(lambda c: c.update(cutoff_len=2048))
    def test_best_checkpoint(self): self.reject_config(lambda c: c.update(load_best_model_at_end=True))
    def test_dev_eval_dataset(self): self.reject_config(lambda c: c.update(eval_dataset='dev'))
    def test_quantization(self): self.reject_config(lambda c: c.update(quantization_bit=4))
    def test_extra_adapter_list(self): self.reject_config(lambda c: c.update(adapter_name_or_path=[s.START]))
    def test_target_modules(self): self.reject_config(lambda c: c.update(lora_target='q_proj'))
    def test_steps_override(self): self.reject_config(lambda c: c.update(max_steps=300))
    def test_hyperparameter(self): self.reject_config(lambda c: c.update(learning_rate=2e-5))
    def test_preflight_source(self):
        self.assertEqual(len(self.indices), 80)
        self.assertEqual(s.read(s.plan.OUT / 'preflight-selection.json')['episode_ids'], [self.episodes[i]['id'] for i in self.indices])

    def test_prepare_only_training_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            records = t.prepare_inputs('preflight', Path(directory))
            rows = s.read(Path(directory) / 'data/train.json')
            self.assertEqual(rows, [self.episodes[i]['row'] for i in self.indices])
            self.assertEqual(records, [self.records[i] for i in self.indices])
            self.assertFalse(s.read(Path(directory) / 'sample.json')['preflight_weights_reused'])

    def test_formal_inputs_all_1200(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(t.prepare_inputs('train', Path(directory)), self.records)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_plan_immutable_binding(self): self.assertTrue(s.plan_gate()['passed'])
    def test_no_execution_from_confirmation(self):
        a = s.acceptance()
        self.assertTrue(a['full_recipe_accepted'])
        self.assertFalse(a['training_authorized'])
        self.assertFalse(a['gpu_inference_authorized'])

    def test_implicit_execution_refused(self):
        for message in (None, '确认', '接受', '授权启动训练'):
            with self.assertRaises(ValueError): s.authorize_record(message)

    def test_reload_adapter_string(self):
        def arguments(**kw):
            self.assertIsInstance(kw['adapter_name_or_path'], str)
            kw['adapter_name_or_path'].split(',')
            return kw
        result = t.reload_arguments(Path('adapter'), arguments)
        self.assertEqual(result['flash_attn'], 'sdpa')

    def test_wrong_adapter_features(self):
        base = {'r': 8, 'lora_alpha': 16, 'lora_dropout': .05, 'target_modules': list(s.old.TARGETS),
                'peft_type': 'LORA', 'task_type': 'CAUSAL_LM'}
        s.check_adapter(base)
        for key, value in [('use_dora', True), ('use_rslora', True), ('r', 16), ('modules_to_save', ['lm_head'])]:
            with self.assertRaises(ValueError): s.check_adapter({**base, key: value})


class DatasetAndTrainerTests(unittest.TestCase):
    def setUp(self):
        self.dataset = [{'input_ids': [1, 2, 3], 'labels': [-100, -100, 3]}]
        self.records = [{'id': 'mock-training-only', 'total_tokens': 3, 'assistant_turns': 1,
                         'input_ids_sha256': s.digest([1, 2, 3]), 'labels_sha256': s.digest([-100, -100, 3])}]

    def test_dataset_exact(self): self.assertEqual(s.check_dataset(self.dataset, self.records)['rows'], 1)
    def test_dataset_row_lost(self):
        with self.assertRaises(ValueError): s.check_dataset([], self.records)
    def test_dataset_truncation(self):
        with self.assertRaises(ValueError): s.check_dataset([{'input_ids': [1, 2], 'labels': [-100, -100]}], self.records)
    def test_user_supervised(self):
        self.dataset[0]['labels'][0] = 1
        with self.assertRaises(ValueError): s.check_dataset(self.dataset, self.records)
    def test_assistant_masked(self):
        self.dataset[0]['labels'][2] = -100
        with self.assertRaises(ValueError): s.check_dataset(self.dataset, self.records)
    def test_tool_return_supervised(self):
        self.dataset[0]['labels'][1] = 2
        with self.assertRaises(ValueError): s.check_dataset(self.dataset, self.records)
    def test_permission_input_changed(self):
        self.dataset[0]['input_ids'][0] = 4
        with self.assertRaises(ValueError): s.check_dataset(self.dataset, self.records)
    def test_optimizer_fresh(self):
        t.initial_gate(NS(global_step=0, max_steps=300), NS(resume_from_checkpoint=None), NS(state={}), NS(last_epoch=0), 300)
    def test_optimizer_inherited(self):
        with self.assertRaises(ValueError):
            t.initial_gate(NS(global_step=0, max_steps=300), NS(resume_from_checkpoint=None), NS(state={'old': 1}), NS(last_epoch=0), 300)
    def test_scheduler_inherited(self):
        with self.assertRaises(ValueError):
            t.initial_gate(NS(global_step=0, max_steps=300), NS(resume_from_checkpoint=None), NS(state={}), NS(last_epoch=150), 300)
    def test_wrong_initial_steps(self):
        with self.assertRaises(ValueError):
            t.initial_gate(NS(global_step=0, max_steps=299), NS(resume_from_checkpoint=None), NS(state={}), NS(last_epoch=0), 300)
    def test_resume_in_callback(self):
        with self.assertRaises(ValueError):
            t.initial_gate(NS(global_step=0, max_steps=300), NS(resume_from_checkpoint='old'), NS(state={}), NS(last_epoch=0), 300)
    def test_state_complete(self):
        s.check_state({'global_step': 300, 'max_steps': 300, 'epoch': 2, 'log_history': [{'loss': .1}]}, {'train_loss': .2}, 300, 2)
    def test_state_insufficient_steps(self):
        with self.assertRaises(ValueError): s.check_state({'global_step': 299, 'max_steps': 300, 'epoch': 2}, {'train_loss': .2}, 300, 2)
    def test_state_insufficient_epoch(self):
        with self.assertRaises(ValueError): s.check_state({'global_step': 300, 'max_steps': 300, 'epoch': 1}, {'train_loss': .2}, 300, 2)
    def test_nonfinite_train_loss(self):
        with self.assertRaises(ValueError): s.check_state({'global_step': 300, 'max_steps': 300, 'epoch': 2}, {'train_loss': float('nan')}, 300, 2)
    def test_nonfinite_log_loss(self):
        with self.assertRaises(ValueError):
            s.check_state({'global_step': 300, 'max_steps': 300, 'epoch': 2, 'log_history': [{'loss': float('inf')}]}, {'train_loss': .2}, 300, 2)
    def test_tensor_initial_mismatch(self):
        with patch.dict(sys.modules, {'torch': FAKE_TORCH}), self.assertRaises(ValueError):
            s.compare_tensor_sets({'m.lora_A.weight': Tensor([1])}, {'m.lora_A.default.weight': Tensor([2])})
    def test_tensor_initial_match(self):
        with patch.dict(sys.modules, {'torch': FAKE_TORCH}):
            self.assertTrue(s.compare_tensor_sets({'m.lora_A.weight': Tensor([1])}, {'m.lora_A.default.weight': Tensor([1])})['passed'])
    def test_updated_tensors(self):
        with patch.dict(sys.modules, {'torch': FAKE_TORCH}):
            self.assertEqual(t.check_updated_tensors({'a': Tensor([1])}, {'a': Tensor([2])})['updated_tensor_count'], 1)
    def test_same_tensors_rejected(self):
        with patch.dict(sys.modules, {'torch': FAKE_TORCH}), self.assertRaises(ValueError):
            t.check_updated_tensors({'a': Tensor([1])}, {'a': Tensor([1])})
    def test_nonfinite_weight(self):
        with patch.dict(sys.modules, {'torch': FAKE_TORCH}), self.assertRaises(ValueError):
            t.check_updated_tensors({'a': Tensor([1])}, {'a': Tensor([float('nan')])})
    def test_missing_weight_keys(self):
        with self.assertRaises(ValueError): t.check_updated_tensors({'a': Tensor([1])}, {})
    def test_shape_changed(self):
        with patch.dict(sys.modules, {'torch': FAKE_TORCH}), self.assertRaises(ValueError):
            t.check_updated_tensors({'a': Tensor([1])}, {'a': Tensor([2, 3])})

    def make_callback(self, directory):
        transformers = ModuleType('transformers')
        transformers.TrainerCallback = object
        safetensors, safetensors_torch = ModuleType('safetensors'), ModuleType('safetensors.torch')
        safetensors_torch.load_file = lambda *args, **kw: {'m.lora_A.weight': Tensor([1])}
        self.fake_modules = {'transformers': transformers, 'safetensors': safetensors,
                             'safetensors.torch': safetensors_torch, 'torch': FAKE_TORCH}
        with patch.dict(sys.modules, self.fake_modules):
            return t.build_callback(Path(directory), self.records, 10)

    def test_callback_actual_inputs_written(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = self.make_callback(directory)
            param = Tensor([1])
            param.requires_grad = True
            with patch.dict(sys.modules, self.fake_modules):
                callback.on_train_begin(NS(resume_from_checkpoint=None), NS(global_step=0, max_steps=10), None,
                    optimizer=NS(state={}), lr_scheduler=NS(last_epoch=0),
                    model=NS(named_parameters=lambda: [('m.lora_A.default.weight', param)]),
                    train_dataloader=FakeLoader(self.dataset))
            self.assertTrue(s.read(Path(directory) / 'initialization.json')['scheduler_fresh'])
            self.assertEqual(s.read(Path(directory) / 'training-input-audit.json')['rows'], 1)

    def test_callback_nonfinite_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = self.make_callback(directory)
            param = NS(requires_grad=True, grad=Tensor([float('inf')]))
            with patch.dict(sys.modules, self.fake_modules), self.assertRaises(ValueError):
                callback.on_pre_optimizer_step(None, None, None, model=NS(named_parameters=lambda: [('a', param)]))

    def test_callback_nonfinite_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = self.make_callback(directory)
            with self.assertRaises(ValueError): callback.on_log(None, None, None, logs={'loss': float('nan')})

    def test_checkpoints_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            for step in (150, 300):
                p = adapter / f'checkpoint-{step}'
                p.mkdir()
                s.write(p / 'trainer_state.json', {'global_step': step, 'epoch': step / 150})
                for name in ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt'): (p / name).write_bytes(b'mock-checkpoint')
            self.assertEqual(set(t.checkpoint_gate(adapter, 'train')), {'checkpoint-150', 'checkpoint-300'})

    def test_checkpoint_missing(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError): t.checkpoint_gate(Path(directory), 'train')


class FakeLoader:
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)


class RuntimeAndEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('repair_launcher_test', s.ROOT / 'scripts/run_ticket_policy_repair_training_v1.py')
        cls.launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.launch)
        # Local project has the hash-locked DEV source descriptor, not the
        # server's approved corpus. Mock only the full remote corpus gate.
        cls.descriptor = s.old.frozen_descriptor()

    def setUp(self):
        self.dev_mock = patch.object(s, 'dev_gate', return_value=(self.descriptor['tasks'], [],
                                      {'scope': 'mock_remote_corpus_gate_for_CPU_tests_only'}))
        self.dev_mock.start()
        self.addCleanup(self.dev_mock.stop)

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError): s.fresh(Path(directory))
    def test_missing_active_launch(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(s, 'RUN', Path(directory)), self.assertRaises(ValueError):
            s.active_launch_gate()
    def test_failed_active_launch(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(s, 'RUN', Path(directory)):
            (Path(directory) / 'launch.lock').write_text('mock-owned-test-lock', encoding='utf-8')
            s.write(Path(directory) / 'status.json', {'status': 'failed'})
            with self.assertRaises(ValueError): s.active_launch_gate()
    def test_wrong_plan_hash(self):
        with patch.object(s, 'sha', return_value='wrong'), self.assertRaises(ValueError): s.plan_gate()
    def test_low_disk_space(self):
        with patch.object(Path, 'stat', return_value=NS(st_size=1024)), patch.object(s.shutil, 'disk_usage', return_value=NS(free=100)), self.assertRaises(ValueError):
            s.storage_gate()
    def test_environment_clean(self):
        clean, removed = s.diagnostic.child_environment({'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0', 'PMI_RANK': '1', 'KEEP': 'yes'})
        s.diagnostic.environment_gate(clean)
        self.assertEqual(removed, ['LOCAL_RANK', 'PMI_RANK', 'RANK', 'WORLD_SIZE'])
        self.assertEqual(clean['KEEP'], 'yes')
        self.assertNotIn('WORLD_SIZE', clean)
    def test_distributed_environment_rejected(self):
        clean, _ = s.diagnostic.child_environment({})
        with self.assertRaises(ValueError): s.diagnostic.environment_gate({**clean, 'WORLD_SIZE': '1'})
    def test_online_rejected(self):
        clean, _ = s.diagnostic.child_environment({})
        with self.assertRaises(ValueError): s.diagnostic.environment_gate({**clean, 'HF_HUB_OFFLINE': '0'})
    def test_gpu_busy_rejected(self):
        with self.assertRaises(ValueError): s.old.check_gpu_rows('GPU-test,A100,81920,80000', 'GPU-test,123,other')
    def test_gpu_insufficient_memory(self):
        with self.assertRaises(ValueError): s.old.check_gpu_rows('GPU-test,A100,81920,50000', '')
    def test_gpu_wrong_hardware(self):
        with self.assertRaises(ValueError): s.old.check_gpu_rows('GPU-test,V100,81920,80000', '')
    def test_gpu_uuid_mismatch(self):
        with self.assertRaises(ValueError): s.old.check_actual_gpu_uuid('GPU-one', 'GPU-two')
    def test_pipeline_sequence(self):
        sequence = self.launch.stage_sequence('all')
        self.assertEqual(len(sequence), 11)
        self.assertEqual([key for _, _, key in sequence[:2]], ['preflight', 'train'])
        for i, stage in enumerate(('smoke', 'dev', 'diagnostic')):
            self.assertEqual([r[2] for r in sequence[2+i*3:5+i*3]],
                             [stage + '-ticket-sft-v3', stage + '-ticket-policy-repair-v1', stage + '-comparison'])
    def test_wrong_variants(self):
        with self.assertRaises(ValueError): e.model_args('hermes-r8')
    def test_baseline_adapter(self): self.assertEqual(e.model_args(s.VARIANTS[0])['adapter_name_or_path'], s.START)
    def test_new_final_adapter(self): self.assertEqual(e.model_args(s.VARIANTS[1])['adapter_name_or_path'], str((s.RUN / 'train/adapter').resolve()))
    def test_smoke_population(self): self.assertEqual(len(e.population('smoke')[0]), 12)
    def test_frozen_dev_population(self): self.assertEqual(len(e.population('dev')[0]), 60)
    def test_diagnostic_population(self): self.assertEqual(len(e.population('diagnostic')[0]), 72)
    def test_only_expected_stage_names(self):
        with self.assertRaises(ValueError): e.population('test')

    def test_failed_model_cases_stay_denominator_and_replay(self):
        rows, _, _, runner, summary = e.population('dev')
        class FailedPolicyProvider:
            def prompt_info(self, payload): return {'tokens': 1, 'kind': 'synthetic_test_not_HF', 'token_ids_sha256': '0'*64}
            def generate(self, payload): return {'text': 'Unable to complete the request.', 'finish_reason': 'stop', 'prompt_tokens': 1, 'response_tokens': 8}
        records = [runner.run_episode(task, FailedPolicyProvider()) for task in rows]
        replayed = [runner.replay_record(task, record) for task, record in zip(rows, records)]
        metric = summary(replayed)
        self.assertEqual((metric['generated_rows'], metric['positive_rows'], metric['negative_rows']), (60, 46, 14))
        self.assertEqual(metric['task_success']['correct'], 0)

    def run_fixture(self):
        rows = [{'id': 'mock-eval'}]
        current = {'final_adapter_sha256': 'a'*64}
        initial = {'mock-eval': {'tokens': 1, 'kind': 'actual_qwen_token_ids', 'token_ids_sha256': 'b'*64}}
        run = {'status': 'complete', 'stage': 'smoke', 'variant': s.VARIANTS[0], 'provider_kind': 'huggingface',
               'versions': s.STACK, 'policy': s.POLICY, 'model_args': e.model_args(s.VARIANTS[0]), 'case_ids': ['mock-eval'],
               'tasks_sha256': s.digest(rows), 'frozen_dev': {'hash': 'frozen'}, 'evidence': current,
               'adapter_sha256': s.START_SHA, 'initial_prompt_audit': initial, 'reference_context_audit_sha256': 'refs',
               'allocated_gpu': {'selection': '1', 'uuid': 'GPU-test', 'name': 'A100'},
               'human_review': 'pending', 'semantic_review': 'pending'}
        return run, rows, current, initial
    def reject_run(self, change):
        run, rows, current, initial = self.run_fixture()
        change(run)
        with self.assertRaises(ValueError): e.check_run(run, 'smoke', s.VARIANTS[0], rows, {'hash': 'frozen'}, current, initial, 'refs')
    def test_eval_run_exact(self):
        run, rows, current, initial = self.run_fixture()
        e.check_run(run, 'smoke', s.VARIANTS[0], rows, {'hash': 'frozen'}, current, initial, 'refs')
    def test_eval_wrong_weights(self): self.reject_run(lambda r: r.update(adapter_sha256='wrong'))
    def test_eval_wrong_training_source(self): self.reject_run(lambda r: r.update(evidence={}))
    def test_eval_missing_cases(self): self.reject_run(lambda r: r.update(case_ids=[]))
    def test_eval_prompt_change(self): self.reject_run(lambda r: r.update(initial_prompt_audit={}))
    def test_eval_policy_change(self): self.reject_run(lambda r: r.update(policy={**s.POLICY, 'max_new_tokens': 1024}))
    def test_eval_fake_human_approval(self): self.reject_run(lambda r: r.update(human_review='approved'))
    def test_eval_wrong_version(self): self.reject_run(lambda r: r.update(versions={}))
    def test_eval_wrong_gpu(self): self.reject_run(lambda r: r.update(allocated_gpu={'selection': '0', 'uuid': 'GPU-test'}))
    def test_eval_run_interrupted(self): self.reject_run(lambda r: r.update(status='failed'))
    def test_change_ids_keep_regressions(self):
        def row(eligible, task, safe): return {'scores': {'metric_eligible': eligible, 'task_success': task, 'safe_behavior': safe}}
        indexed = {s.VARIANTS[0]: {'a': row(True, True, False), 'b': row(False, False, False)},
                   s.VARIANTS[1]: {'a': row(True, False, False), 'b': row(False, False, True)}}
        changes = e.change_ids(indexed)
        self.assertEqual(changes['regressed_ids'], ['a'])
        self.assertEqual(changes['negative_safe_behavior_improved_ids'], ['b'])
    def test_smoke_no_quality_gate(self):
        with patch.object(e, 'replay', return_value=({}, {'all_scores': 0}, {})), patch.object(s, 'read', return_value={'run_sha256': {v: 'same' for v in s.VARIANTS}, 'replay_passed': True}), patch.object(s, 'sha', return_value='same'):
            e.prerequisite('dev')
    def test_sequential_gpu_and_time(self):
        gpu = {'selection': '1', 'uuid': 'GPU-test', 'name': 'A100'}
        runs = {s.VARIANTS[0]: {'allocated_gpu': gpu, 'finished_at_utc': '2026-10-07T10:00:00+00:00'},
                s.VARIANTS[1]: {'allocated_gpu': gpu, 'started_at_utc': '2026-10-07T10:01:00+00:00'}}
        e.check_sequential(runs)
        runs[s.VARIANTS[1]]['started_at_utc'] = '2026-10-07T09:00:00+00:00'
        with self.assertRaises(ValueError): e.check_sequential(runs)


if __name__ == '__main__': unittest.main(verbosity=2)
