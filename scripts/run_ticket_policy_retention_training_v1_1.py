"""Fresh, locked GPU1 stages; CPU verify needs no execution authorization."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.ticket_policy_retention_training_v1_1 import support as s


def stage_sequence(stage):
    s.require(stage in ('all', 'preflight', 'train', 'evaluate'), 'Unknown execution stage')
    sequence = []
    if stage in ('all', 'preflight'): sequence.append(('train', ['--stage', 'preflight'], 'preflight'))
    if stage in ('all', 'train'): sequence.append(('train', ['--stage', 'train'], 'train'))
    if stage in ('all', 'evaluate'):
        for phase in tuple(s.COUNTS):
            for variant in s.VARIANTS:
                sequence.append(('evaluate', ['--stage', phase, '--variant', variant], phase + '-' + variant))
            sequence.append(('evaluate', ['--stage', phase, '--compare'], phase + '-comparison'))
    return sequence


def step(module, arguments, key):
    log = s.RUN / 'logs' / (key + '.log')
    status_path = log.with_suffix('.status.json')
    for path in (log, status_path): s.fresh(path)
    log.parent.mkdir(parents=True, exist_ok=True)
    environment, removed = s.diagnostic.child_environment(os.environ)
    s.diagnostic.environment_gate(environment)
    environment.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    command = [sys.executable, '-u', '-B', '-m', 'extensions.ticket_policy_retention_training_v1_1.' + module, *arguments]
    record = {'status': 'started', 'started_at_utc': s.now(), 'command': command, 'gpu': '1',
              'removed_environment_keys': removed, 'distributed_environment_absent': True,
              'controlled_environment': {k: environment[k] for k in ('CUDA_VISIBLE_DEVICES', 'FORCE_TORCHRUN', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')},
              'sources': s.sources(), 'authorization_sha256': s.execution_gate()}
    s.write(status_path, record)
    process = None
    try:
        with log.open('x', encoding='utf-8') as stream:
            print('Launching:', command, flush=True)
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                stream.write(line)
                stream.flush()
            code = process.wait()
        s.require(code == 0, f'Stage failed (exit={code}); preserve log and output: {log}')
        record.update(status='complete', exit_code=code)
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()  # only our child, never another GPU task
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['finished_at_utc'] = s.now()
        if log.exists(): record['log_sha256'] = s.sha(log)
        s.update(status_path, record)


def final_report():
    from extensions.ticket_policy_retention_training_v1_1.train import check_completed
    check_completed('preflight'); check_completed('train')
    path = s.RUN / 'final-report.json'; s.fresh(path); s.fresh(path.with_suffix('.md'))
    comparisons = {phase:s.read(s.eval_directory(phase)/'comparison.json') for phase in s.COUNTS}
    for phase,report in comparisons.items():
        s.require(report['replay_passed'] and report['actual_all_round_tokenizer_reaudit']
            and report['rows_per_variant']==s.COUNTS[phase], 'Incomplete paired evidence')
    dev = comparisons['dev']; targeted = comparisons['targeted']['metrics'][s.VARIANTS[1]]
    mechanical = not dev['regressed_ids'] and targeted['constraint_violations']==0 and targeted['task_success']['correct']==192
    files = ['execution-authorization.json','preflight/run.json','preflight/verification.json','train/run.json','train/verification.json']
    files += [f'eval/{phase}/comparison.json' for phase in s.COUNTS]
    proof={'status':'training_and_paired_evaluation_complete','training':s.read(s.RUN/'train/verification.json'),
        'generated_evaluation_episodes':672,'comparisons':comparisons,
        'source_files_sha256':{name:s.sha(s.RUN/name) for name in files},
        'mechanical_quality_gate_passed':mechanical,'dev_new_positive_regression_ids':dev['regressed_ids'],
        'semantic_review':'pending','human_review':'pending','model_quality_release_passed':None,
        'model_quality_release_status':'pending_semantics' if mechanical else 'blocked_mechanical_findings',
        'ai_review_is_human_approval':False,'is_generalization_evidence':False,'old_final_test_read':False,
        'sources':s.sources(),'created_at_utc':s.now()}
    s.write(path,proof)
    with path.with_suffix('.md').open('x',encoding='utf-8') as stream:
        stream.write('# 能力保留续训与配对评测交接\n\n1800 条／5202 个助手轮次，225 步／1 epoch；真实保存重载见 train/verification.json。\n\n'
            '四组评测共 672 次 episode；12 条检查与 DEV 重叠，不作为独立证据。最终语义待审核，AI 审查不是人工批准。\n\n'
            '工程完成不代表模型质量通过。机械门禁：'+str(mechanical)+'；模型质量发布尚未批准。\n\n'
            '60 条仍为冻结 DEV，72 条仍为诊断，192 条为已知风险针对性开发评测；不是最终测试或泛化证明。\n')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('verify', 'all', 'preflight', 'train', 'evaluate'), required=True)
    parser.add_argument('--execution-authorization', help='Only after separate user authorization: 授权启动训练及配对评测')
    args = parser.parse_args()
    os.chdir(ROOT)
    verified = s.delivery_gate()  # CPU only; no CUDA/model/optimizer import
    if args.stage == 'verify':
        import json
        if s.old.DEV.is_dir():
            verified['remote_full_dev_gate'] = s.dev_gate()[2]
        else:
            verified['remote_full_dev_gate'] = 'Corpus unavailable locally; mandatory gate still required before any training/inference'
        print(json.dumps(verified, ensure_ascii=False, indent=2))
        return
    s.require(sys.version_info[:2] == (3, 11), 'Use the pinned server Python 3.11')
    authorization = s.authorize_record(args.execution_authorization)
    if args.stage in ('preflight', 'all'):
        s.fresh(s.RUN)
        s.RUN.mkdir(parents=True)
        s.write(s.RUN / 'execution-authorization.json', authorization)
    else:
        s.require(s.RUN.is_dir(), 'Run preflight first; no automatic recovery')
        s.execution_gate()
        status = s.read(s.RUN / 'status.json')
        expected = 'preflight' if args.stage == 'train' else 'train'
        s.require(status['status'] == 'complete' and status['last_completed_stage'] == expected,
                  'Run failed/incomplete or wrong order; do not overwrite/resume')
    lock = s.RUN / 'launch.lock'
    with lock.open('x', encoding='utf-8') as stream: stream.write(str(os.getpid()))
    status = {'status': 'started', 'requested_stage': args.stage, 'started_at_utc': s.now(),
              'authorization_sha256': s.execution_gate(), 'last_completed_stage': None}
    s.update(s.RUN / 'status.json', status)
    try:
        for module, arguments, key in stage_sequence(args.stage):
            step(module, arguments, key)
            status['last_completed_stage'] = key
            s.update(s.RUN / 'status.json', status)
        if args.stage in ('all', 'evaluate'): final_report()
        status.update(status='complete', finished_at_utc=s.now())
        s.update(s.RUN / 'status.json', status)
        print('Requested stages completed; final semantics pending. Frozen DEV remains DEV.', flush=True)
    except BaseException as exc:
        status.update(status='failed', finished_at_utc=s.now(), error=f'{type(exc).__name__}: {exc}')
        s.update(s.RUN / 'status.json', status)
        raise
    finally:
        lock.unlink()  # exact owned lock only, all logs/failure outputs retained


if __name__ == '__main__': main()


