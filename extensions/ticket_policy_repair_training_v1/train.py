"""Real Trainer evidence; separate from every previous/hash-bound training module."""
import gc
from pathlib import Path
from . import support as s


def initial_gate(state, args, optimizer, scheduler, steps):
    s.require(state.global_step == 0 and state.max_steps == steps and args.resume_from_checkpoint is None,
              'Wrong step budget or checkpoint/optimizer resume')
    s.require(not optimizer.state, 'Inherited optimizer state')
    s.require(scheduler is not None and scheduler.last_epoch in (-1, 0), 'Inherited/missing scheduler state')


def build_callback(directory, records, steps):
    from transformers import TrainerCallback
    from safetensors.torch import load_file
    import torch

    class Evidence(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            initial_gate(state, args, kw['optimizer'], kw['lr_scheduler'], steps)
            trainable = {k: p for k, p in kw['model'].named_parameters() if p.requires_grad}
            s.require(trainable and all('.lora_A.' in k or '.lora_B.' in k for k in trainable), 'Base/new non-LoRA parameters trainable')
            result = s.compare_tensor_sets(load_file(str(Path(s.START) / 'adapter_model.safetensors'), device='cpu'), trainable)
            result.update(initializer_sha256=s.START_SHA, optimizer_fresh=True, scheduler_fresh=True, global_step=0)
            s.write(directory / 'initialization.json', result)
            loader = kw['train_dataloader']
            s.require(len(loader) == len(records), 'Unexpected batch count/drop_last')
            s.write(directory / 'training-input-audit.json', s.check_dataset(loader.dataset, records))
            s.write(directory / 'reload-input.json', {'source_id': records[0]['id'],
                    'input_ids': list(loader.dataset[0]['input_ids'])[:32], 'scope': 'Training-only finite reload forward'})
            print(f'Actual Trainer input/labels verified: {len(records)} rows', flush=True)

        def on_pre_optimizer_step(self, args, state, control, **kw):
            for name, param in kw['model'].named_parameters():
                if param.requires_grad and param.grad is not None:
                    s.require(torch.isfinite(param.grad).all().item(), 'Nonfinite gradient: ' + name)

        def on_log(self, args, state, control, logs=None, **kw):
            for key in ('loss', 'train_loss', 'grad_norm', 'learning_rate'):
                if logs and key in logs:
                    s.require(s.finite(logs[key]), 'Nonfinite ' + key)

    return Evidence()


def prepare_inputs(stage, directory):
    episodes, records, indices, longest = s.inputs()
    if stage == 'train':
        return records
    data = directory / 'data'
    s.fresh(data)
    data.mkdir()
    s.write(data / 'train.json', [episodes[i]['row'] for i in indices])
    registration = s.read(s.RELEASE / 'dataset_info.json')[s.plan.release.DATASET]
    s.write(data / 'dataset_info.json', {'ticket_policy_repair_preflight_v1': registration})
    s.write(directory / 'sample.json', {'seed': 42, 'rows': 80, 'pairs': 40,
            'ids': [episodes[i]['id'] for i in indices], 'longest_pair_by_category': longest,
            'train_sha256': s.sha(data / 'train.json'), 'release_manifest_sha256': s.RELEASE_SHA,
            'formal_initializer': s.START, 'preflight_weights_reused': False})
    return [records[i] for i in indices]


def check_updated_tensors(initial, final):
    s.require(initial and set(initial) == set(final), 'Saved adapter tensor keys differ')
    import torch
    changed = []
    for key in initial:
        s.require(final[key].shape == initial[key].shape and torch.isfinite(final[key]).all().item(), 'Wrong shape/nonfinite adapter: ' + key)
        if not torch.equal(initial[key].to(dtype=final[key].dtype), final[key]):
            changed.append(key)
    s.require(changed, 'Adapter tensors unchanged; byte hash alone is not a weight update')
    return {'tensor_count': len(final), 'updated_tensor_count': len(changed), 'updated_keys': changed, 'all_tensors_finite': True}


def checkpoint_gate(adapter, stage):
    expected = (10,) if stage == 'preflight' else (150, 300)
    directories = {p.name: p for p in adapter.glob('checkpoint-*') if p.is_dir()}
    s.require(set(directories) == {f'checkpoint-{i}' for i in expected}, 'Missing/unexpected epoch checkpoint')
    proof = {}
    for step in expected:
        directory = directories[f'checkpoint-{step}']
        state = s.read(directory / 'trainer_state.json')
        s.require(state['global_step'] == step and s.finite(state['epoch'])
                  and abs(state['epoch'] - (1 if stage == 'preflight' else step / 150)) < 1e-6, 'Checkpoint step/epoch differs')
        for name in ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json'):
            s.require((directory / name).is_file() and (directory / name).stat().st_size > 0, 'Incomplete checkpoint: ' + name)
        proof[directory.name] = {p.name: s.sha(p) for p in sorted(directory.iterdir()) if p.is_file()}
    return proof


def verify_result(directory, stage):
    adapter = directory / 'adapter'
    steps, rows, turns = (10, 80, None) if stage == 'preflight' else (300, 1200, 2560)
    s.check_state(s.read(adapter / 'trainer_state.json'), s.read(adapter / 'all_results.json'), steps,
                  1.0 if stage == 'preflight' else 2.0)
    s.check_adapter(s.read(adapter / 'adapter_config.json'))
    weights = adapter / 'adapter_model.safetensors'
    s.require(weights.is_file() and weights.stat().st_size > 0 and s.sha(weights) != s.START_SHA, 'Weights missing/empty/unchanged')
    audit, initial = s.read(directory / 'training-input-audit.json'), s.read(directory / 'initialization.json')
    all_records = s.inputs()[1]
    records = all_records if stage == 'train' else [all_records[i] for i in s.inputs()[2]]
    s.require(audit == {'passed': True, 'rows': rows, 'assistant_turns': sum(r['assistant_turns'] for r in records),
                        'ordered_audit_sha256': s.digest(records)}, 'Actual training input evidence differs')
    s.require(turns is None or audit['assistant_turns'] == turns, 'Assistant turns lost')
    s.require(initial['passed'] is True and initial['optimizer_fresh'] is True and initial['scheduler_fresh'] is True
              and initial['initializer_sha256'] == s.START_SHA and initial['global_step'] == 0, 'Initialization evidence differs')
    s.require(s.sha(Path(s.START) / 'adapter_model.safetensors') == s.START_SHA
              and s.sha(Path(s.START) / 'adapter_config.json') == s.START_CONFIG_SHA, 'Original initializer modified')
    from safetensors.torch import load_file
    changed = check_updated_tensors(load_file(str(Path(s.START) / 'adapter_model.safetensors'), device='cpu'),
                                    load_file(str(weights), device='cpu'))
    return {'passed': True, 'rows': rows, 'assistant_turns': audit['assistant_turns'], 'global_step': steps,
            'epoch': 1 if stage == 'preflight' else 2, 'train_loss': s.read(adapter / 'all_results.json')['train_loss'],
            'adapter_sha256': s.sha(weights), 'adapter_config_sha256': s.sha(adapter / 'adapter_config.json'),
            'adapter_bytes': weights.stat().st_size, 'initializer_unchanged': True, **changed,
            'checkpoint_files_sha256': checkpoint_gate(adapter, stage)}


def reload_arguments(adapter, argument_class):
    return argument_class(model_name_or_path=s.MODEL, adapter_name_or_path=str(adapter.resolve()),
                          trust_remote_code=False, flash_attn='sdpa')


def reload_check(adapter, first_ids):
    from llamafactory.hparams import ModelArguments, FinetuningArguments
    from llamafactory.model import load_model, load_tokenizer
    from safetensors.torch import load_file
    import torch
    args = reload_arguments(adapter, ModelArguments)  # dataclass expects string, not list
    args.compute_dtype = torch.bfloat16
    tokenizer = load_tokenizer(args)['tokenizer']
    model = load_model(tokenizer, args, FinetuningArguments(finetuning_type='lora', create_new_adapter=False), is_trainable=True)
    model.to(torch.device('cuda:0'))
    result = s.compare_tensor_sets(load_file(str(adapter / 'adapter_model.safetensors'), device='cpu'),
                                  {k: p for k, p in model.named_parameters() if p.requires_grad})
    model.eval()
    s.require(first_ids and len(first_ids) <= 32, 'Missing training-only reload input')
    ids = torch.tensor([first_ids], dtype=torch.long, device=next(model.parameters()).device)
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        s.require(torch.isfinite(output.logits).all().item(), 'Reloaded forward nonfinite')
    result.update(finite_forward=True, adapter_sha256=s.sha(adapter / 'adapter_model.safetensors'))
    del model, output, ids
    gc.collect()
    torch.cuda.empty_cache()
    return result


def check_completed(stage):
    directory = s.RUN / stage
    run = s.read(directory / 'run.json')
    s.require(run['status'] == 'complete' and run['stage'] == stage and run['sources'] == s.sources()
              and run['dependencies'] == s.dependencies() and run['config'] == s.config_for(stage)
              and run['authorization_sha256'] == s.execution_gate()
              and run['release_manifest_sha256'] == s.RELEASE_SHA and run['repair_initializer_sha256'] == s.START_SHA,
              'Prerequisite incomplete/changed; never resume/relabel failed work')
    expected_names = {'adapter/adapter_config.json', 'adapter/adapter_model.safetensors', 'adapter/trainer_state.json',
                      'adapter/all_results.json', 'initialization.json', 'training-input-audit.json',
                      'reload-input.json', 'reload.json', 'verification.json', 'config.json'}
    if stage == 'preflight': expected_names |= {'sample.json', 'data/train.json', 'data/dataset_info.json'}
    s.require(set(run['outputs_sha256']) == expected_names, 'Training output evidence incomplete')
    for name, expected in run['outputs_sha256'].items():
        s.require(s.sha(directory / name) == expected, 'Training evidence changed: ' + name)
    result = verify_result(directory, stage)
    reload = s.read(directory / 'reload.json')
    s.require(reload['passed'] is True and reload['finite_forward'] is True
              and reload['adapter_sha256'] == result['adapter_sha256'], 'Reload missing/wrong adapter')
    s.require(s.read(directory / 'verification.json') == {**result, 'reload': reload}, 'Saved acceptance differs')
    return run


def execute(stage):
    s.delivery_gate()
    s.active_launch_gate()
    authorization = s.execution_gate()
    config = s.read(s.config_path(stage))
    s.check_config(config, stage)
    directory = s.RUN / stage
    s.fresh(directory)
    if stage == 'train': check_completed('preflight')
    storage = s.storage_gate()
    versions, gpu = s.runtime_gate()
    artifacts = s.artifact_gate()
    dev_meta = s.dev_gate()[2]
    directory.mkdir()
    run = {'status': 'started', 'stage': stage, 'started_at_utc': s.now(), 'versions': versions, 'allocated_gpu': gpu,
           'storage_admission': storage,
           'config': config, 'config_sha256': s.sha(s.config_path(stage)), 'sources': s.sources(), 'dependencies': s.dependencies(),
           'release_manifest_sha256': s.RELEASE_SHA, 'authorization_sha256': authorization, **artifacts, **dev_meta,
           'initialization': 'Current ticket-sft-v3 weights; fresh optimizer/scheduler; never preflight weights',
           'selection': 'Fixed two epochs/final adapter; no DEV loss or best-checkpoint selection'}
    s.write(directory / 'run.json', run)
    try:
        s.write(directory / 'config.json', config)
        records = prepare_inputs(stage, directory)
        from transformers import set_seed
        from llamafactory.hparams import get_train_args
        from llamafactory.train.sft.workflow import run_sft
        set_seed(42)
        args = get_train_args(config)
        s.require(args[2].resume_from_checkpoint is None, 'No optimizer/checkpoint resume')
        run_sft(*args, callbacks=[build_callback(directory, records, 10 if stage == 'preflight' else 300)])
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        result = verify_result(directory, stage)
        reload = reload_check(directory / 'adapter', s.read(directory / 'reload-input.json')['input_ids'])
        s.write(directory / 'reload.json', reload)
        s.write(directory / 'verification.json', {**result, 'reload': reload})
        s.require(s.sources() == run['sources'] and s.dependencies() == run['dependencies']
                  and s.artifact_gate() == artifacts and s.dev_gate()[2] == dev_meta, 'Immutable sources changed during training')
        s.plan.release.verify()
        names = ['adapter/adapter_config.json', 'adapter/adapter_model.safetensors', 'adapter/trainer_state.json',
                 'adapter/all_results.json', 'initialization.json', 'training-input-audit.json', 'reload-input.json',
                 'reload.json', 'verification.json', 'config.json']
        if stage == 'preflight': names += ['sample.json', 'data/train.json', 'data/dataset_info.json']
        run.update(status='complete', finished_at_utc=s.now(), outputs_sha256={n: s.sha(directory / n) for n in names})
        s.update(directory / 'run.json', run)
        print({**result, 'reload': reload}, flush=True)
    except BaseException as exc:
        run.update(status='failed', finished_at_utc=s.now(), error=f'{type(exc).__name__}: {exc}')
        s.update(directory / 'run.json', run)
        raise


if __name__ == '__main__':
    import argparse
    import os
    import signal
    def interrupted(signum, frame): raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight', 'train'), required=True)
    os.chdir(s.ROOT)
    execute(parser.parse_args().stage)
