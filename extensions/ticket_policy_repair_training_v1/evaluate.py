"""Paired original DEV/diagnostic prompts, real HF inference and all-round replay."""
from datetime import datetime
from pathlib import Path
from . import support as s
from .train import check_completed


def model_args(variant):
    s.require(variant in s.VARIANTS, 'Unknown adapter variant')
    return {'model_name_or_path': s.MODEL, 'adapter_name_or_path': s.START if variant == s.VARIANTS[0]
            else str((s.RUN / 'train/adapter').resolve()), 'template': 'qwen', 'infer_backend': 'huggingface',
            'finetuning_type': 'lora', 'trust_remote_code': False, 'infer_dtype': 'bfloat16', 'flash_attn': 'sdpa',
            'do_sample': False, 'max_new_tokens': 512, 'repetition_penalty': 1.0}


def population(stage):
    s.require(stage in ('smoke', 'dev', 'diagnostic'), 'Unknown evaluation stage')
    tasks, refs, dev_meta = s.dev_gate()
    if stage == 'diagnostic':
        rows = s.diagnostic.build_tasks()
        s.diagnostic.validate_tasks(rows, tasks)
        runner = s.diagnostic.isolated_agent()
        summary = lambda records: s.diagnostic.summary(records, rows)
    else:
        rows = s.legacy().selected_tasks(tasks, stage)
        s.legacy()
        import agent
        runner, summary = agent, agent.summarize
    s.require(s.POLICY == runner.POLICY == s.legacy().POLICY, 'Generation/replay policy differs')
    s.require(len(rows) == {'smoke': 12, 'dev': 60, 'diagnostic': 72}[stage], 'Missing/filtered evaluation population')
    return rows, refs, dev_meta, runner, summary


def token_provider():
    from llamafactory.hparams import ModelArguments, DataArguments
    from llamafactory.model import load_tokenizer
    from llamafactory.data import get_template_and_fix_tokenizer
    tokenizer = load_tokenizer(ModelArguments(model_name_or_path=s.MODEL, trust_remote_code=False))['tokenizer']
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template='qwen'))
    return s.legacy().HFProvider(tokenizer=tokenizer, template=template)


def initial_prompts(provider, rows, runner):
    result = {t['id']: provider.prompt_info(runner.input_payload([{'role': 'user', 'content': t['request']}],
                                                              runner.system_prompt(t))) for t in rows}
    s.require(all(i['tokens'] + 512 <= 8192 for i in result.values()), 'Context overflow; do not remove cases')
    return result


def evidence():
    check_completed('train')
    return {'artifacts': s.artifact_gate(), 'training_run_sha256': s.sha(s.RUN / 'train/run.json'),
            'final_adapter_sha256': s.sha(s.RUN / 'train/adapter/adapter_model.safetensors'),
            'final_adapter_config_sha256': s.sha(s.RUN / 'train/adapter/adapter_config.json'),
            'release_manifest_sha256': s.RELEASE_SHA, 'authorization_sha256': s.execution_gate(),
            'sources': s.sources(), 'dependencies': s.dependencies()}


def check_run(run, stage, variant, rows, dev_meta, current, initial, reference_hash):
    s.require(run['status'] == 'complete' and run['stage'] == stage and run['variant'] == variant
              and run['provider_kind'] == 'huggingface' and run['versions'] == s.STACK
              and run['policy'] == s.POLICY and run['model_args'] == model_args(variant), 'Eval runtime/configuration mismatch')
    s.require(run['case_ids'] == [t['id'] for t in rows] and run['tasks_sha256'] == s.digest(rows)
              and run['frozen_dev'] == dev_meta and run['evidence'] == current, 'Eval population/source mismatch')
    expected_sha = s.START_SHA if variant == s.VARIANTS[0] else current['final_adapter_sha256']
    s.require(run['adapter_sha256'] == expected_sha and run['initial_prompt_audit'] == initial
              and run['reference_context_audit_sha256'] == reference_hash, 'Wrong adapter/prompts/reference context')
    s.require(run['allocated_gpu']['selection'] == '1' and run['allocated_gpu']['uuid'].startswith('GPU-'), 'Wrong GPU')
    s.require(run['semantic_review'] == run['human_review'] == 'pending', 'Unreviewed semantics mislabelled')


def check_sequential(runs):
    old, new = (runs[v] for v in s.VARIANTS)
    for key in ('selection', 'uuid', 'name'):
        s.require(old['allocated_gpu'][key] == new['allocated_gpu'][key], 'Paired GPU identity differs')
    s.require(datetime.fromisoformat(old['finished_at_utc']) <= datetime.fromisoformat(new['started_at_utc']), 'Not old-then-new sequential')


def replay(stage):
    rows, refs, dev_meta, runner, summary = population(stage)
    current = evidence()
    provider = token_provider()  # actual tokenizer only; no model weights/GPU inference
    initial = initial_prompts(provider, rows, runner)
    reference_hash = s.digest(s.legacy().audit_reference_contexts(provider, refs)) if stage != 'diagnostic' else None
    runs, metrics, indexed = {}, {}, {}
    for variant in s.VARIANTS:
        directory = s.eval_directory(stage, variant)
        run = s.read(directory / 'run.json')
        check_run(run, stage, variant, rows, dev_meta, current, initial, reference_hash)
        s.require(set(run['outputs_sha256']) == {'tasks.json', 'episodes.jsonl', 'metrics.json', 'answer-review.json'}, 'Eval hashes incomplete')
        for name, expected in run['outputs_sha256'].items():
            s.require(s.sha(directory / name) == expected, 'Evaluation output changed: ' + name)
        s.require(s.read(directory / 'tasks.json') == rows, 'Saved evaluation requests changed')
        saved = [s.diagnostic.strict_json(line) for line in (directory / 'episodes.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        s.require([r['id'] for r in saved] == [t['id'] for t in rows], 'Case dropped/duplicated/reordered')
        records = [runner.replay_record(t, r) for t, r in zip(rows, saved)]
        for r in saved:
            s.require(r['rounds'] and r['rounds'][0]['prompt_info'] == initial[r['id']], 'Wrong initial input')
            for step in r['rounds']:
                s.require(provider.prompt_info(step['input']) == step['prompt_info'], 'Real all-round token history changed')
        metrics[variant] = summary(records)
        s.require(metrics[variant] == s.read(directory / 'metrics.json'), 'Scores differ from independent replay')
        if stage == 'dev':
            s.require(metrics[variant]['positive_rows'] == 46 and metrics[variant]['negative_rows'] == 14, 'DEV denominators changed')
        runs[variant], indexed[variant] = run, {r['id']: r for r in records}
    check_sequential(runs)
    return runs, metrics, indexed


def change_ids(indexed):
    a, b = (indexed[v] for v in s.VARIANTS)
    s.require(list(a) == list(b), 'Paired denominators differ')
    eligible = [i for i in a if a[i]['scores']['metric_eligible']]
    negative = [i for i in a if not a[i]['scores']['metric_eligible']]
    def changes(ids, key, improved):
        return [i for i in ids if bool(a[i]['scores'][key]) == (not improved) and bool(b[i]['scores'][key]) == improved]
    return {'improved_ids': changes(eligible, 'task_success', True), 'regressed_ids': changes(eligible, 'task_success', False),
            'negative_safe_behavior_improved_ids': changes(negative, 'safe_behavior', True),
            'negative_safe_behavior_regressed_ids': changes(negative, 'safe_behavior', False)}


def comparison(stage):
    output = s.eval_directory(stage) / 'comparison.json'
    s.fresh(output)
    runs, metrics, indexed = replay(stage)
    changes = change_ids(indexed)
    report = {'stage': stage, 'split': 'post_DEV_diagnostic' if stage == 'diagnostic' else 'dev',
              'rows_per_variant': {'smoke': 12, 'dev': 60, 'diagnostic': 72}[stage], 'metrics': metrics, **changes,
              'run_sha256': {v: s.sha(s.eval_directory(stage, v) / 'run.json') for v in s.VARIANTS},
              'replay_passed': True, 'actual_all_round_tokenizer_reaudit': True,
              'training_run_sha256': s.sha(s.RUN / 'train/run.json'), 'release_manifest_sha256': s.RELEASE_SHA,
              'semantic_review': 'pending', 'human_review': 'pending', 'ai_review_is_human_approval': False,
              'scope': 'Frozen DEV and previously used diagnostic only; no generalization, speedup or final-test claim',
              'coverage_limits': s.plan.recipe()['evaluation']['coverage_limits']}
    if stage != 'diagnostic':
        a, b = (metrics[v]['task_success'] for v in s.VARIANTS)
        report.update(task_success_delta_correct=b['correct'] - a['correct'],
                      task_success_delta_percentage_points=100 * (b['rate'] - a['rate']))
    s.write(output, report)
    text = '# 工单修复配对评测：' + stage + '\n\n'
    text += '完整独立重放及每轮真实 tokenizer 输入核对通过。模型失败保留在分母，语义与人工审核仍待完成。\n\n'
    for v in s.VARIANTS:
        text += v + '：\n\n```json\n' + s.diagnostic.canonical(metrics[v]) + '\n```\n\n'
    text += '改善／退步 ID：\n\n```json\n' + s.diagnostic.canonical(changes) + '\n```\n\n'
    text += '仅为冻结 DEV 与既有诊断，不是独立最终测试、泛化证明或速度收益。澄清续接、空结果、部分成功覆盖仍有限。\n'
    with output.with_suffix('.md').open('x', encoding='utf-8') as stream: stream.write(text)
    print(report, flush=True)
    return report


def prerequisite(stage):
    previous = {'dev': 'smoke', 'diagnostic': 'dev'}.get(stage)
    if previous:
        runs, _, _ = replay(previous)  # infrastructure/provenance only; no score threshold
        report = s.read(s.eval_directory(previous) / 'comparison.json')
        s.require(report['run_sha256'] == {v: s.sha(s.eval_directory(previous, v) / 'run.json') for v in s.VARIANTS}
                  and report['replay_passed'] is True, 'Previous comparison incomplete/changed')


def execute(stage, variant):
    s.delivery_gate()
    s.active_launch_gate()
    s.execution_gate()
    output = s.eval_directory(stage, variant)
    s.fresh(output)
    prerequisite(stage)
    if variant == s.VARIANTS[1]:
        s.require(s.read(s.eval_directory(stage, s.VARIANTS[0]) / 'run.json')['status'] == 'complete', 'Evaluate baseline first')
    versions, gpu = s.runtime_gate()
    current = evidence()
    rows, refs, dev_meta, runner, summary = population(stage)
    provider = token_provider()
    initial = initial_prompts(provider, rows, runner)
    ref_hash = s.digest(s.legacy().audit_reference_contexts(provider, refs)) if stage != 'diagnostic' else None
    output.mkdir(parents=True)
    run = {'status': 'started', 'stage': stage, 'variant': variant, 'provider_kind': 'huggingface', 'versions': versions,
           'started_at_utc': s.now(), 'allocated_gpu': gpu, 'model_args': model_args(variant), 'policy': s.POLICY,
           'case_ids': [t['id'] for t in rows], 'tasks_sha256': s.digest(rows), 'evidence': current, 'frozen_dev': dev_meta,
           'adapter_sha256': s.START_SHA if variant == s.VARIANTS[0] else current['final_adapter_sha256'],
           'initial_prompt_audit': initial, 'reference_context_audit_sha256': ref_hash,
           'semantic_review': 'pending', 'human_review': 'pending'}
    s.write(output / 'run.json', run)
    try:
        s.write(output / 'tasks.json', rows)
        (output / 'progress').mkdir()
        from transformers import set_seed
        from llamafactory.chat import ChatModel
        set_seed(42)
        model = ChatModel(model_args(variant))
        provider = s.legacy().HFProvider(model)
        s.require(initial_prompts(provider, rows, runner) == initial, 'Loaded engine changed input tokenization')
        records = []
        with (output / 'episodes.jsonl').open('x', encoding='utf-8') as stream:
            for i, task in enumerate(rows):
                record = runner.run_episode(task, provider, progress_path=output / 'progress' / (task['id'] + '.json'))
                runner.replay_record(task, record)
                s.require(record['rounds'][0]['prompt_info'] == initial[task['id']], 'Actual initial prompt changed')
                stream.write(s.diagnostic.canonical(record) + '\n')
                stream.flush()
                records.append(record)
                print(f'{stage} {variant} {i+1}/{len(rows)}: {task["id"]} -> {record["termination"]}', flush=True)
        s.write(output / 'metrics.json', summary(records))
        s.write(output / 'answer-review.json', [{'id': r['id'], 'request': t['request'], 'answer': r['final_text'],
                'profile': t.get('prompt_profile'), 'scores': r['scores'], 'human_review': 'pending',
                'semantic_review': 'pending', 'ai_review': 'not_started'} for t, r in zip(rows, records)])
        s.require(evidence() == current and population(stage)[2] == dev_meta, 'Sources changed during inference')
        run.update(status='complete', finished_at_utc=s.now(), outputs_sha256={n: s.sha(output / n) for n in
                   ('tasks.json', 'episodes.jsonl', 'metrics.json', 'answer-review.json')})
        s.update(output / 'run.json', run)
    except BaseException as exc:
        run.update(status='failed', finished_at_utc=s.now(), error=f'{type(exc).__name__}: {exc}')
        s.update(output / 'run.json', run)
        raise


if __name__ == '__main__':
    import argparse
    import os
    import signal
    def interrupted(signum, frame): raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('smoke', 'dev', 'diagnostic'), required=True)
    parser.add_argument('--variant', choices=s.VARIANTS)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    s.require(args.compare != bool(args.variant), 'Choose one variant or comparison')
    os.chdir(s.ROOT)
    s.diagnostic.environment_gate(os.environ)
    s.delivery_gate()
    s.execution_gate()
    s.active_launch_gate()
    comparison(args.stage) if args.compare else execute(args.stage, args.variant)
