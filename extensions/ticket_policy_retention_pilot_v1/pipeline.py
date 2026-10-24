"""Append-only representative build/review and verification, never bulk or train."""
import argparse
import html
import json
import sys
from . import core as c
from .validate import forbidden, validate_population


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def sources():
    files = sorted(c.HERE.glob('*.py')) + [c.HERE/'README.md',
            c.ROOT/'scripts/run_ticket_policy_retention_pilot_v1.py',
            c.ROOT/'scripts/package_ticket_policy_retention_pilot_v1.py',
            c.ROOT/'tests/test_ticket_policy_retention_pilot_v1.py']
    return {p.relative_to(c.ROOT).as_posix():c.sha(p) for p in files}


def gate():
    from scripts.verify_ticket_policy_retention_plan_v1 import inspect
    result = inspect()
    c.require(result['passed'], 'Confirmed plan/source gate failed')
    c.require(not any(n.split('.')[0] in {'torch','transformers','llamafactory'} for n in sys.modules), 'Model framework imported in CPU-only pilot')
    return result


def isolation_record(blocked):
    return {k:blocked[k] for k in ('frozen_dev_sha256','old_train_sha256','diagnostic_tasks_sha256','previous_repair_episodes_sha256')} | {
        'forbidden_identifiers_count':len(blocked['ids']), 'forbidden_requests_count':len(blocked['requests']),
        'forbidden_identifiers_sha256':c.digest(sorted(blocked['ids'])),
        'forbidden_requests_sha256':c.digest(sorted(blocked['requests'])),
        'exact_or_identity_normalized_request_overlap':0, 'identifier_overlap':0,
        'semantic_template_disjointness_proven':False,
        'scope':'DEV, previous training and development diagnostics only; no old final test. Shared concepts/schema are not unique identifiers.'}


def review(episodes):
    lines = ['# 策略修复与能力保留：20 对／40 条代表 episode','',
             '仅离线模拟正确策略，没有模型生成或训练。AI 辅助审核与用户接受分别记录；待确认，不可直接训练。',
             '数据库变化须区分模型授权写入和外部故障事件。完整 JSON 中的 fixture、claims 和验证元数据不进入模型输入。','']
    cards = []
    for zh,en in zip(episodes[::2],episodes[1::2]):
        s = zh['spec']
        heading = f'{s["pair_id"]} · {s["theme"]}/{s["variant"]} · {s["permission_profile"]} · allow_write={s["allow_write"]}'
        lines += ['## '+heading,'','审核目标：'+s['requirement'],'']
        body = [f'<h2>{html.escape(heading)}</h2><p>审核目标：{html.escape(s["requirement"])}</p>']
        for label,e in [('中文',zh),('English',en)]:
            chain = ' → '.join(x['call']['name']+(' ✓' if x['response']['ok'] else ' ✗ '+x['response']['error']['code']) for x in e['events']) or '无调用 / no calls'
            visible = '\n'.join(m['from']+': '+m['value'] for m in e['row']['conversations'] if m['from'] in ('human','gpt'))
            lines += ['### '+label,'','调用链：'+chain,'','```text',visible,'```','']
            body += [f'<h3>{label}</h3><p>{html.escape(chain)}</p><pre>{html.escape(visible)}</pre>',
                     f'<details><summary>{label}完整模型可见输入（系统、工具与全部轮次）</summary><pre>{html.escape(json.dumps(e["row"],ensure_ascii=False,indent=2))}</pre></details>']
        changes = {'initial':zh['initial_state'],'final':zh['final_state'],'external_events':zh['external_events']}
        writes = sum(x['call']['name'] in c.WRITE_TOOLS for x in zh['events'])
        state_note = f'写工具提案 {writes} 次，外部事件 {len(zh["external_events"])} 次；数据库'+('有变化。' if zh['initial_state']!=zh['final_state'] else '未改变。')
        lines += [state_note,'','```json',json.dumps(changes,ensure_ascii=False,indent=2),'```','']
        body += [f'<p>{state_note}</p><details><summary>初始／最终状态与外部事件</summary><pre>{html.escape(json.dumps(changes,ensure_ascii=False,indent=2))}</pre></details>',
                 f'<details><summary>完整中英 episode JSON（离线元数据不进入模型）</summary><pre>{html.escape(json.dumps([zh,en],ensure_ascii=False,indent=2))}</pre></details>']
        cards.append('<section>'+''.join(body)+'</section>')
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>能力保留代表样例</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}section{border-top:1px solid #aaa;padding:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:14px}details{margin:12px 0}</style><h1>20 对／40 条代表样例</h1><p>真实 SQLite 模拟返回；不是模型成绩、人工批准或训练发布。AI 审核结论单列。</p>'+''.join(cards)+'</html>'
    return '\n'.join(lines)+'\n', page


def fresh():
    c.require(not c.OUT.exists() and not c.REPORT.exists(), 'Existing output; refuse overwrite/resume. Preserve it and diagnose.')


def build():
    verified = gate()
    fresh()
    c.OUT.mkdir(parents=True)
    c.REPORT.mkdir(parents=True)
    write(c.REPORT/'user-design-confirmation.json',{
        'date':'2026-10-08','user_message':'确认','accepted_plan_sha256':c.PLAN_SHA,
        'representative_generation_authorized':True,'bulk_generation_authorized':False,
        'training_authorized':False,'gpu_inference_authorized':False,
        'provenance':'Explicit current-chat user confirmation of design; not sample acceptance, per-row inspection or verified signature'})
    write(c.OUT/'status.json',{'status':'started','training_started':False})
    try:
        episodes = [c.generate(s,l) for s in c.specs() for l in ('zh','en')]
        write(c.OUT/'episodes.json',episodes)
        # Preview rows only, not registered as a trainable released dataset.
        write(c.OUT/'sharegpt-preview.json',[e['row'] for e in episodes])
        blocked = forbidden()
        validation = validate_population(episodes,blocked)
        write(c.OUT/'validation.json',validation)
        write(c.OUT/'isolation.json',isolation_record(blocked))
        md,page = review(episodes)
        (c.OUT/'review.md').write_text(md,encoding='utf-8')
        (c.OUT/'review.html').write_text(page,encoding='utf-8')
        names = ('episodes.json','sharegpt-preview.json','validation.json','isolation.json','review.md','review.html')
        manifest = {'version':c.VERSION,'seed':c.SEED,'rows':40,'pairs':20,
                    'confirmed_plan_sha256':c.PLAN_SHA,'sources_sha256':sources(),
                    'simulator_bridge_sha256':c.SIM_SHA,'simulator_v1_sha256':c.sim.V1_SHA,
                    'files_sha256':{n:c.sha(c.OUT/n) for n in names},
                    'confirmation_sha256':c.sha(c.REPORT/'user-design-confirmation.json'),
                    'design_gate':verified,'ready_for_training':False,'sample_acceptance':'pending',
                    'real_tokenizer_audit':'pending_remote','bulk_generated':False,
                    'training_started':False,'gpu_inference_started':False,'old_final_test_read':False}
        write(c.OUT/'manifest.json',manifest)
        write(c.OUT/'status.json',{'status':'complete','manifest_sha256':c.sha(c.OUT/'manifest.json'),
                                  'sample_acceptance':'pending','ready_for_training':False,'training_started':False})
        print(json.dumps({k:v for k,v in validation.items() if k!='records'},ensure_ascii=False,indent=2))
    except BaseException as error:
        write(c.OUT/'status.json',{'status':'failed','error':repr(error),'training_started':False})
        (c.REPORT/'failure.log').write_text(__import__('traceback').format_exc(),encoding='utf-8')
        raise


def verify():
    gate()
    manifest = c.read(c.OUT/'manifest.json')
    c.require(manifest['sources_sha256']==sources() and manifest['confirmed_plan_sha256']==c.PLAN_SHA
              and manifest['simulator_bridge_sha256']==c.SIM_SHA and manifest['simulator_v1_sha256']==c.sim.V1_SHA,
              'Source binding changed')
    c.require(manifest['ready_for_training'] is False and manifest['sample_acceptance']=='pending'
              and manifest['real_tokenizer_audit']=='pending_remote'
              and manifest['bulk_generated'] is False and manifest['training_started'] is False
              and manifest['gpu_inference_started'] is False and manifest['old_final_test_read'] is False,
              'Review/training gate overstated')
    c.require(set(manifest['files_sha256'])=={'episodes.json','sharegpt-preview.json','validation.json','isolation.json','review.md','review.html'}, 'Missing artifact binding')
    for name,expected in manifest['files_sha256'].items():
        c.require('/' not in name and '\\' not in name and c.sha(c.OUT/name)==expected,'Artifact hash differs: '+name)
    c.require(c.sha(c.REPORT/'user-design-confirmation.json')==manifest['confirmation_sha256'], 'User confirmation changed')
    status = c.read(c.OUT/'status.json')
    c.require(status['status']=='complete' and status['manifest_sha256']==c.sha(c.OUT/'manifest.json'), 'Build incomplete')
    blocked = forbidden()
    result = validate_population(c.read(c.OUT/'episodes.json'),blocked)
    c.require(result==c.read(c.OUT/'validation.json') and isolation_record(blocked)==c.read(c.OUT/'isolation.json'), 'Replay/isolation changed')
    c.require(c.read(c.OUT/'sharegpt-preview.json')==[e['row'] for e in c.read(c.OUT/'episodes.json')], 'Preview rows differ')
    return {'passed':True,'rows':40,'pairs':20,'independent_replay_rows':40,'sample_acceptance':'pending',
            'ready_for_training':False,'real_tokenizer_audit':'pending_remote','training_started':False,
            'gpu_inference_started':False,'old_final_test_read':False,'manifest_sha256':c.sha(c.OUT/'manifest.json')}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('build','verify'))
    args = parser.parse_args()
    if args.command=='build':
        build()
    else:
        print(json.dumps(verify(),ensure_ascii=False,indent=2))
