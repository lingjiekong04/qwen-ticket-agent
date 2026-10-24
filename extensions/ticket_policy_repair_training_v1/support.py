"""Separate repair run: immutable recipe, append-only evidence and runtime gates."""
import json
import os
import shutil
from pathlib import Path
from extensions.ticket_policy_repair_training_plan_v1 import core as plan
from extensions.ticket_training_v1_2 import support as old
from extensions.ticket_policy_diagnostic_v1_1 import core as diagnostic

ROOT = old.ROOT
HERE = Path(__file__).resolve().parent
REPORT = ROOT / 'reports/ticket-policy-repair-training-v1'
RUN = ROOT / plan.NEW_RUN
RELEASE = plan.release.DATA
RELEASE_SHA = plan.recipe()['data']['release_manifest_sha256']
START = plan.START
START_SHA = plan.START_SHA
START_CONFIG_SHA = plan.START_CONFIG_SHA
MODEL = old.MODEL
STACK = old.STACK
VARIANTS = ('ticket-sft-v3', 'ticket-policy-repair-v1')
POLICY = plan.recipe()['evaluation']['policy']
for _name in ('require', 'sha', 'digest', 'read', 'write', 'update', 'now', 'fresh', 'finite',
              'check_adapter', 'check_state', 'check_dataset', 'compare_tensor_sets', 'legacy', 'dev_gate'):
    globals()[_name] = getattr(old, _name)


def config_for(stage):
    require(stage in ('preflight', 'train'), 'Unknown training stage')
    config = old.predecessor.config_for(stage)
    config.update(adapter_name_or_path=START, learning_rate=1e-5,
                  lora_target=','.join(sorted(old.TARGETS)),
                  output_dir=f'{plan.NEW_RUN}/{stage}/adapter',
                  dataset_dir=RELEASE.relative_to(ROOT).as_posix(), dataset=plan.release.DATASET)
    if stage == 'preflight':
        config.update(dataset_dir=f'{plan.NEW_RUN}/preflight/data', dataset='ticket_policy_repair_preflight_v1')
    return config


def config_path(stage):
    require(stage in ('preflight', 'train'), 'Unknown configuration')
    return ROOT / f'configs/ticket_policy_repair_{stage}_v1.yaml'


def check_config(config, stage):
    require(config == config_for(stage), 'Configuration changed: no mixing/resume/new LoRA/DEV/extra flags permitted')


def sources():
    files = list(HERE.glob('*.py')) + [HERE / 'README.md']
    files += [config_path(stage) for stage in ('preflight', 'train')]
    files += [ROOT / 'scripts' / name for name in ('run_ticket_policy_repair_training_v1.py',
              'package_ticket_policy_repair_training_v1.py')]
    files += [ROOT / 'tests/test_policy_repair_training_v1.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def dependencies():
    # Explicit approved namespaces only. Never enumerate old/final-test data.
    files = [ROOT / name for name in plan.s.read(plan.OUT / 'validation.json')['source_bindings']]
    files += [plan.HERE / 'core.py', plan.HERE / 'PLAN.md',
              ROOT / 'extensions/ticket_training_v1_1/support.py', ROOT / 'extensions/ticket_training_v1_1/train.py',
              ROOT / 'extensions/ticket_training_v1_2/support.py',
              ROOT / 'extensions/ticket_policy_diagnostic_v1/core.py',
              ROOT / 'extensions/ticket_policy_diagnostic_v1_1/core.py']
    files += [(ROOT / 'extensions/ticket_agent_v2' / name).resolve() for name in legacy().source_hashes()]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}


def plan_gate():
    validation = read(plan.OUT / 'validation.json')
    require(sha(plan.OUT / 'validation.json') == '9e90c0c90a6d4650812b4bc1c98e8b900faec3759bd01772eec03c15e67e727f',
            'Historical plan validation changed')
    require(read(plan.OUT / 'status.json')['status'] == 'complete', 'Historical proposal incomplete')
    for name, expected in validation['files_sha256'].items():
        require(sha(plan.OUT / name) == expected, 'Historical plan evidence changed: ' + name)
    for name, expected in {**validation['source_bindings'], **validation['implementation_sources']}.items():
        require(sha(ROOT / name) == expected, 'Plan source changed: ' + name)
    plan.validate(read(plan.OUT / 'plan.json'))
    return validation


def acceptance():
    return {'date': '2026-10-07', 'user_message': '确认', 'full_recipe_accepted': True,
            'launcher_implementation_authorized': True, 'training_authorized': False,
            'training_started': False, 'gpu_inference_authorized': False, 'gpu_inference_started': False,
            'scope': 'Accept the complete 1200-row plan and implement/deliver the incremental launcher only; execution needs separate authorization',
            'provenance': 'Explicit current-chat user confirmation; not independently authenticated identity/signature',
            'plan_sha256': sha(plan.OUT / 'plan.json'), 'plan_validation_sha256': sha(plan.OUT / 'validation.json')}


def inputs():
    episodes = read(plan.release.c.OUT / 'episodes.json')
    records = [json.loads(line) for line in (RELEASE / 'token-records.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = read(RELEASE / 'train.json')
    require(len(rows) == len(records) == len(episodes) == 1200, 'Missing training rows')
    require(rows == [e['row'] for e in episodes], 'Dataset differs from released episodes')
    for e, r in zip(episodes, records):
        require(e['id'] == r['id'] and digest(e['row']) == r['row_sha256'] and r['passed'] is True,
                'Episode/token evidence differs')
    indices, longest = plan.select_preflight(episodes, records)
    selection = read(plan.OUT / 'preflight-selection.json')
    require(selection['episode_ids'] == [episodes[i]['id'] for i in indices]
            and selection['longest_pair_by_category'] == longest
            and selection['subset_rows_digest'] == digest([rows[i] for i in indices]), 'Preflight selection changed')
    require(sum(r['assistant_turns'] for r in records) == 2560, 'Assistant denominator differs')
    return episodes, records, indices, longest


def local_gate():
    plan_gate()
    result = plan.release.verify()
    require(result['release_manifest_sha256'] == RELEASE_SHA, 'Wrong repair release')
    require(read(REPORT / 'full-plan-acceptance.json') == acceptance(), 'Complete plan acceptance missing/changed')
    for stage in ('preflight', 'train'):
        # JSON is valid YAML; stdlib reading keeps CPU verification dependency-free.
        check_config(read(config_path(stage)), stage)
    inputs()
    descriptor = old.frozen_descriptor()
    tasks = descriptor['tasks']
    dev_meta = {'frozen_tasks_sha256': old.DEV_SHA, 'definition_and_fixture_hash_checked': True,
                'server_corpus_and_approval': 'pending_remote_full_dev_gate',
                'scope': 'Local frozen source descriptor only; not a substitute for remote data/approval validation'}
    probes = diagnostic.build_tasks()
    diagnostic.validate_tasks(probes, tasks)
    baseline = read(plan.DIAGNOSTIC_RUN)
    require([t['id'] for t in probes] == baseline['case_ids'], 'Existing diagnostic population changed')
    return {'passed': True, 'rows': 1200, 'pairs': 600, 'assistant_turns': 2560,
            'expected_steps': 300, 'planned_evaluation_episodes': 288, 'frozen_dev': dev_meta,
            'diagnostic_tasks_sha256': digest(probes), 'release_manifest_sha256': RELEASE_SHA,
            'training_authorized': False, 'training_started': False, 'gpu_inference_started': False,
            'remote_runtime_and_weights': 'pending_server_execution', 'old_final_test_read': False}


def delivery_gate():
    manifest = read(REPORT / 'delivery.json')
    require(manifest['training_authorized'] is False and manifest['training_started'] is False
            and manifest['gpu_inference_started'] is False, 'Delivery overstates execution')
    require(manifest['sources'] == sources() and manifest['dependencies'] == dependencies(), 'Delivery source changed')
    for name, expected in manifest['report_files_sha256'].items():
        require(Path(name).name == name and sha(REPORT / name) == expected, 'Delivery evidence changed')
    return local_gate()


def artifact_gate():
    expected = read(plan.DIAGNOSTIC_RUN)['artifact_evidence']
    actual = old.artifact_gate()  # CPU hashing/source checks; no model load.
    for key in ('choice_sha256', 'initializer_sha256', 'initializer_config_sha256',
                'base_weights_sha256', 'framework_source_sha256'):
        require(actual[key] == expected[key], 'Old immutable base/framework evidence differs: ' + key)
    # Also verifies the previous training acceptance without running its launcher.
    from extensions.ticket_training_v1_1.train import check_completed
    check_completed('train')
    previous = ROOT / 'runs/ticket-sft-v3-hermes-r8-v1.1/train/run.json'
    require(sha(previous) == expected['training_run_sha256'], 'Wrong initial training run')
    require(sha(Path(START) / 'adapter_model.safetensors') == START_SHA
            and sha(Path(START) / 'adapter_config.json') == START_CONFIG_SHA, 'Wrong current ticket-sft-v3 initializer')
    check_adapter(read(Path(START) / 'adapter_config.json'))
    audit = read(RELEASE / 'token-audit.json')
    for name, expected_hash in audit['tokenizer_asset_sha256'].items():
        require(Path(name).name == name and sha(Path(MODEL) / name) == expected_hash, 'Repair tokenizer asset differs')
    import inspect
    from llamafactory.data.converter import SharegptDatasetConverter
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams import get_train_args
    source = Path(inspect.getsourcefile(get_train_args)).resolve()
    pinned_root = ROOT / 'LlamaFactory-ca75f1edf3cb50343ed1c98605141c3e22075b5f/src'
    require(source.is_relative_to(pinned_root.resolve()), 'Framework is not the pinned checkout')
    for cls in (SharegptDatasetConverter, SupervisedDatasetProcessor):
        require(sha(inspect.getsourcefile(cls)) == audit['framework_source_sha256'][cls.__name__], 'Repair preprocessor source differs')
    return {**actual, 'repair_initializer_sha256': START_SHA, 'repair_initializer_config_sha256': START_CONFIG_SHA,
            'initial_training_run_sha256': sha(previous)}


def runtime_gate():
    diagnostic.environment_gate(os.environ)
    return old.runtime_gate()


def storage_gate():
    # Two LoRA checkpoints with optimizer/scheduler plus final weights and logs.
    # Capacity admission only; never trim checkpoints or change the recipe.
    required = max(2 * 1024**3, (Path(START) / 'adapter_model.safetensors').stat().st_size * 16 + 256 * 1024**2)
    free = shutil.disk_usage(ROOT).free
    require(free >= required, 'Insufficient output disk space; stop without reducing data/checkpoints')
    return {'free_bytes': free, 'required_free_bytes': required, 'kind': 'conservative_preload_capacity_admission'}


def authorize_record(message):
    require(message == '授权启动训练及配对评测', 'Separate explicit execution authorization required')
    return {'user_message': message, 'scope': '10-step preflight, 1200-row 300-step continuation and 288 paired inference episodes',
            'kind': 'user_cli_attestation_not_independent_authentication', 'accepted_plan_sha256': sha(plan.OUT / 'plan.json'),
            'delivery_sha256': sha(REPORT / 'delivery.json'), 'release_manifest_sha256': RELEASE_SHA,
            'initializer_sha256': START_SHA, 'training_authorized': True, 'gpu_inference_authorized': True,
            'created_at_utc': now()}


def execution_gate():
    record = read(RUN / 'execution-authorization.json')
    expected = authorize_record(record['user_message'])
    expected['created_at_utc'] = record['created_at_utc']
    require(record == expected, 'Execution authorization/provenance differs')
    return sha(RUN / 'execution-authorization.json')


def active_launch_gate():
    require((RUN / 'launch.lock').is_file() and read(RUN / 'status.json')['status'] == 'started',
            'Use the authorized fresh-stage launcher; failed/completed runs cannot run a child directly')


def eval_directory(stage, variant=None):
    require(stage in ('smoke', 'dev', 'diagnostic'), 'Unknown evaluation stage')
    require(variant is None or variant in VARIANTS, 'Unknown evaluated adapter')
    return RUN / 'eval' / stage / variant if variant else RUN / 'eval' / stage
