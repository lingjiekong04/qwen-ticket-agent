"""Append-only offline representative generation; AI judgments are authored separately."""
import argparse
import html
import json
import subprocess
import sys
import traceback
from . import core as c
from . import validate as v


def write_new(path,value):
    with path.open('x',encoding='utf-8') as stream: stream.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def text_new(path,value):
    with path.open('x',encoding='utf-8') as stream: stream.write(value)


def sources():
    paths=list(c.HERE.glob('*.py'))+[c.HERE/'README.md',
        c.ROOT/'scripts/run_ticket_policy_retention_extension_v1_1.py',
        c.ROOT/'scripts/package_ticket_policy_retention_extension_v1_1.py',
        c.ROOT/'tests/test_ticket_policy_retention_extension_v1_1.py']
    return {p.relative_to(c.ROOT).as_posix():c.sha(p) for p in sorted(paths)}


def gate():
    from scripts.freeze_ticket_policy_retention_design_v1 import verify
    result=verify()
    c.require(result['freeze_sha256']==c.FREEZE_SHA,'Wrong design freeze')
    from extensions.ticket_policy_retention_extension_v1.pipeline import verify as previous_verify
    previous=previous_verify()
    old_report=c.ROOT/'reports/ticket-policy-retention-extension-v1'
    c.require(previous['manifest_sha256']==c.PREVIOUS_MANIFEST
              and c.sha(old_report/'ai-review.json')==c.PREVIOUS_AI_REVIEW
              and c.sha(old_report/'ai-review-notes.json')==c.PREVIOUS_NOTES,'Original blocked revision evidence changed')
    c.require(not any(n.split('.')[0] in {'torch','transformers','llamafactory'} for n in sys.modules),'GPU/model framework imported')
    return result | {'previous_extension_manifest_sha256':previous['manifest_sha256'],'previous_AI_review_sha256':c.PREVIOUS_AI_REVIEW}


def fresh():
    c.require(not c.OUT.exists() and not c.REPORT.exists(),'Existing output; refuse overwrite or resume')


def isolation(blocked):
    return {'forbidden_identifiers':len(blocked['ids']),'forbidden_requests':len(blocked['requests']),
            'identifiers_sha256':c.digest(sorted(blocked['ids'])),'requests_sha256':c.digest(sorted(blocked['requests'])),
            'exact_or_identity_normalized_request_overlap':0,'identifier_overlap':0,'semantic_template_disjointness_proven':False,
            'old_pilot_included':True,'old_final_test_read':False,'scope':'Previous training, frozen DEV, development diagnostics and accepted pilot only'}


def revision_comparison(episodes):
    previous=c.read(c.ROOT/'data/ticket_policy_retention_extension_v1/episodes.json')
    c.require(len(previous)==len(episodes)==24,'Revision population changed')
    unchanged=[]; changed=[]
    fields=('row','events','claims','initial_state','final_state','environment_audit','external_events')
    for before,after in zip(previous,episodes):
        c.require(before['id']==after['id'],'Revision identity changed')
        same=all(before[key]==after[key] for key in fields)
        if after['spec']['index']!=12:
            c.require(same and before['spec']==after['spec'],'Unaffected representative changed')
            unchanged.append(after['id'])
        else:
            c.require(not same and before['row']['system']==after['row']['system']
                      and before['row']['tools']==after['row']['tools'] and before['row']['conversations'][0]==after['row']['conversations'][0],'Repair altered request/policy or did not change failure')
            changed.append({'id':after['id'],'old_row_sha256':c.digest(before['row']),'new_row_sha256':c.digest(after['row'])})
    return {'passed':True,'previous_manifest_sha256':c.PREVIOUS_MANIFEST,'unchanged_rows':unchanged,'changed_rows':changed,
            'repair':'Second ticket is assignable when read; real external deletion after that read causes ticket_not_found',
            'original_evidence_preserved':True,'old_final_test_read':False}


def review(episodes):
    lines=['# 补充代表：12 对／24 条','',
           '真实隔离 SQLite 返回与重放；不是模型成绩、训练发布或人工批准。工具／用户标签屏蔽仅检查结构，真实 tokenizer 审计待远端。','']
    cards=[]
    for zh,en in zip(episodes[::2],episodes[1::2]):
        s=zh['spec']; title=f'{s["pair_id"]} · {s["variant"]} · {s["permission_profile"]} · allow_write={s["allow_write"]}'
        lines+=['## '+title,'']; card=['<h2>'+html.escape(title)+'</h2>']
        for locale,e in [('中文',zh),('English',en)]:
            chain=' → '.join(x['call']['name']+(' ✓' if x['response']['ok'] else ' ✗ '+x['response']['error']['code']) for x in e['events']) or '无调用 / no calls'
            visible='\n'.join(m['from']+': '+m['value'] for m in e['row']['conversations'] if m['from'] in ('human','gpt'))
            lines+=['### '+locale,'','调用链：'+chain,'','```text',visible,'```','']
            card +=['<h3>'+locale+'</h3><p>'+html.escape(chain)+'</p><pre>'+html.escape(visible)+'</pre>',
                    '<details><summary>完整模型可见输入</summary><pre>'+html.escape(json.dumps(e['row'],ensure_ascii=False,indent=2))+'</pre></details>']
        changes={'initial':zh['initial_state'],'final':zh['final_state'],'external_events':zh['external_events']}
        note=f'写提案 {sum(x["call"]["name"] in c.WRITE_TOOLS for x in zh["events"])} 次，外部事件 {len(zh["external_events"])} 次。'
        lines += [note,'','```json',json.dumps(changes,ensure_ascii=False,indent=2),'```','']
        card +=['<p>'+note+'</p><details><summary>状态变化与故障证据</summary><pre>'+html.escape(json.dumps(changes,ensure_ascii=False,indent=2))+'</pre></details>',
                 '<details><summary>完整中英 JSON（离线元数据不进入模型）</summary><pre>'+html.escape(json.dumps([zh,en],ensure_ascii=False,indent=2))+'</pre></details>']
        cards.append('<section>'+''.join(card)+'</section>')
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>补充代表审核</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}section{border-top:1px solid #aaa;padding:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:14px}details{margin:12px 0}</style><h1>12 对／24 条补充代表</h1><p>模拟正确轨迹；AI 辅助审核与用户接受分开，不是模型成绩或训练发布。</p>'+''.join(cards)+'</html>'
    return '\n'.join(lines)+'\n',page


def build():
    verified=gate(); fresh()
    c.OUT.mkdir(parents=True); c.REPORT.mkdir(parents=True)
    write_new(c.REPORT/'user-generation-confirmation.json',{
        'date':'2026-10-08','user_message':'继续','accepted_scope':'Repair the blocked partial-assignment representative under a fresh revision; preserve original evidence and rerun all 24 representatives with AI-assisted review; no bulk or training authority',
        'design_freeze_sha256':c.FREEZE_SHA,'extension_generation_authorized':True,'bulk_generation_authorized':False,
        'evaluation_task_instantiation_authorized':False,'training_authorized':False,'gpu_inference_authorized':False,
        'sample_acceptance':'pending','per_row_human_inspection_attested':False,
        'provenance':'Explicit current-chat user continuation after scoped confirmation question; not independently verified identity/signature'})
    write_new(c.OUT/'started.json',{'status':'started','training_started':False})
    try:
        run=subprocess.run([sys.executable,'-B','-m','unittest',
            'tests.test_ticket_policy_retention_design_v1','tests.test_ticket_policy_retention_pilot_v1',
            'tests.test_ticket_policy_retention_extension_v1','tests.test_ticket_policy_retention_extension_v1_1','-v'],
            cwd=c.ROOT,capture_output=True,text=True,encoding='utf-8')
        text_new(c.REPORT/'tests.log',run.stdout+run.stderr)
        c.require(run.returncode==0,'Tests failed; preserve diagnostics')
        episodes=[c.generate(s,l) for s in c.specs() for l in ('zh','en')]
        write_new(c.OUT/'episodes.json',episodes)
        write_new(c.OUT/'sharegpt-preview.json',[e['row'] for e in episodes])
        blocked=v.blocked_sources()
        validation=v.validate_population(episodes,blocked)
        write_new(c.OUT/'validation.json',validation); write_new(c.OUT/'isolation.json',isolation(blocked))
        write_new(c.OUT/'revision-comparison.json',revision_comparison(episodes))
        md,page=review(episodes)
        text_new(c.OUT/'review.md',md); text_new(c.OUT/'review.html',page)
        names=('episodes.json','sharegpt-preview.json','validation.json','isolation.json','review.md','review.html','revision-comparison.json')
        manifest={'version':c.VERSION,'seed':c.SEED,'rows':24,'pairs':12,'design_freeze_sha256':c.FREEZE_SHA,
                  'sources_sha256':sources(),'files_sha256':{n:c.sha(c.OUT/n) for n in names},'design_gate':verified,
                  'confirmation_sha256':c.sha(c.REPORT/'user-generation-confirmation.json'),
                  'test_log_sha256':c.sha(c.REPORT/'tests.log'),'ready_for_training':False,
                  'sample_acceptance':'pending','actual_token_label_audit':'pending_remote','bulk_generated':False,
                  'training_started':False,'gpu_inference_started':False,'old_final_test_read':False}
        write_new(c.OUT/'manifest.json',manifest)
        write_new(c.OUT/'status.json',{'status':'complete','manifest_sha256':c.sha(c.OUT/'manifest.json'),'sample_acceptance':'pending','training_started':False})
        result=verify(); write_new(c.REPORT/'mechanical-verification.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except BaseException as error:
        write_new(c.REPORT/'failure.json',{'status':'failed','error':repr(error),'training_started':False})
        text_new(c.REPORT/'failure.log',traceback.format_exc()); raise


def verify():
    gate(); manifest=c.read(c.OUT/'manifest.json')
    c.require(manifest['version']==c.VERSION and manifest['rows']==24 and manifest['pairs']==12 and manifest['seed']==c.SEED,'Population/version changed')
    c.require(manifest['sources_sha256']==sources() and manifest['design_freeze_sha256']==c.FREEZE_SHA,'Sources/design changed')
    for key in ('ready_for_training','bulk_generated','training_started','gpu_inference_started','old_final_test_read'):
        c.require(manifest[key] is False,'Boundary overstated: '+key)
    c.require(manifest['sample_acceptance']=='pending' and manifest['actual_token_label_audit']=='pending_remote','Review/token audit overstated')
    names={'episodes.json','sharegpt-preview.json','validation.json','isolation.json','review.md','review.html','revision-comparison.json'}
    c.require(set(manifest['files_sha256'])==names,'File binding incomplete')
    for name,expected in manifest['files_sha256'].items(): c.require(c.sha(c.OUT/name)==expected,'Artifact changed: '+name)
    confirmation=c.read(c.REPORT/'user-generation-confirmation.json')
    c.require(c.sha(c.REPORT/'user-generation-confirmation.json')==manifest['confirmation_sha256']
              and confirmation['extension_generation_authorized'] is True and all(confirmation[k] is False for k in
              ('bulk_generation_authorized','evaluation_task_instantiation_authorized','training_authorized','gpu_inference_authorized','per_row_human_inspection_attested')),'Confirmation scope changed')
    c.require(c.sha(c.REPORT/'tests.log')==manifest['test_log_sha256'],'Test evidence changed')
    status=c.read(c.OUT/'status.json')
    c.require(status['status']=='complete' and status['manifest_sha256']==c.sha(c.OUT/'manifest.json'),'Build incomplete')
    episodes=c.read(c.OUT/'episodes.json'); blocked=v.blocked_sources()
    validation=v.validate_population(episodes,blocked)
    c.require(validation==c.read(c.OUT/'validation.json') and isolation(blocked)==c.read(c.OUT/'isolation.json'),'Replay/isolation changed')
    c.require(revision_comparison(episodes)==c.read(c.OUT/'revision-comparison.json'),'Revision scope/evidence changed')
    c.require([e['row'] for e in episodes]==c.read(c.OUT/'sharegpt-preview.json'),'Preview differs')
    md,page=review(episodes)
    c.require((c.OUT/'review.md').read_text(encoding='utf-8')==md and (c.OUT/'review.html').read_text(encoding='utf-8')==page,'Review surface differs')
    return {k:v for k,v in validation.items() if k!='records'} | {'manifest_sha256':c.sha(c.OUT/'manifest.json'),
                'episodes_sha256':c.sha(c.OUT/'episodes.json'),'unchanged_representative_rows':22,'repaired_representative_rows':2,
                'previous_blocked_revision_preserved':True,'legacy_bound_files_checked':35,'local_python':sys.version.split()[0]}


def record_review():
    result=verify(); notes=c.read(c.REPORT/'ai-review-notes.json')
    c.require(notes['review_type']=='AI_assisted' and notes['episodes_sha256']==result['episodes_sha256']
              and notes['manifest_sha256']==result['manifest_sha256'],'Notes unbound or misattributed')
    c.require(notes['human_acceptance']=='pending' and notes['per_row_human_inspection'] is False,'AI review marked human')
    c.require([x['pair_id'] for x in notes['pair_notes']]==[s['pair_id'] for s in c.specs()],'Incomplete authored review')
    c.require(all(x['judgment'] in ('no_blocking_issue_found','blocking_issue') for x in notes['pair_notes']),'Unknown review judgment')
    receipt={'scope':'Bind already-authored AI judgments, not an automatic semantic classifier','review_type':'AI_assisted',
             'mechanical_verification':result,'notes_sha256':c.sha(c.REPORT/'ai-review-notes.json'),'human_acceptance':'pending',
             'per_row_human_inspection':False,'blocking_findings':sum(x['judgment']=='blocking_issue' for x in notes['pair_notes']),
             'actual_token_label_audit':'pending_remote','ready_for_training':False,'training_started':False,'gpu_inference_started':False}
    write_new(c.REPORT/'ai-review.json',receipt)
    md=['# 补充 12 对：AI 辅助审核结论','',
        '已逐对阅读中英请求、实际返回、调用依赖、状态变化及最终回答。机械重放与语义判断分开，不是模型评测、逐条人工批准或训练发布。','']
    for note in notes['pair_notes']: md+=['## '+note['pair_id'],'',note['note'],'',('限制：'+note['caution']) if note.get('caution') else '','']
    md+=['## 当前门禁','',
         '24 条均已机械重放。真实 tokenizer／ShareGPT 转换器／标签审计仍待远端；自然中文和英文释义为人工撰写的模板代表，事实子句对齐不证明模型已学会。',
         '用户本轮只授权生成与审核，补充样例接受仍 pending；1800 条批量、192 条实际评测、训练与 GPU 未执行。','']
    text_new(c.REPORT/'ai-review.md','\n'.join(md))
    write_new(c.REPORT/'verification.json',{'passed':True,'manifest_sha256':result['manifest_sha256'],
              'review_files_sha256':{name:c.sha(c.REPORT/name) for name in ('ai-review-notes.json','ai-review.json','ai-review.md')},
              'AI_review':'completed','blocking_findings':receipt['blocking_findings'],'semantic_template_gate': 'clear_pending_user_acceptance' if receipt['blocking_findings']==0 else 'blocked',
              'human_acceptance':'pending','actual_token_label_audit':'pending_remote','ready_for_training':False,'training_started':False})
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=('build','verify','record-review'))
    args=parser.parse_args()
    if args.stage=='build': build()
    elif args.stage=='record-review': record_review()
    else: print(json.dumps(verify(),ensure_ascii=False,indent=2))
