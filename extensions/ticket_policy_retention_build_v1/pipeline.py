"""Append-only confirmation, full targeted-task freeze and candidate construction."""
import argparse
from datetime import datetime,timezone
import html
import json
import subprocess
import sys
import traceback
from . import core as c,validate as v

EXT_MANIFEST='d449448545fbace22e42499e2039d8a02b082498b07293f8d3e4a2ced2ba95a2'
EXT_EPISODES='26adce65467af291013361c1944e77aa890206a0a5d88f4264ebefdcf0cfe9d5'
SUPERVISION={'template':'qwen','cutoff_len':4096,'train_on_prompt':False,'mask_history':False,'packing':False}

def now(): return datetime.now(timezone.utc).isoformat()
def write(path,value):
    with path.open('x',encoding='utf-8') as f: f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def text(path,value):
    with path.open('x',encoding='utf-8') as f: f.write(value)
def fresh(path): c.require(not path.exists(),'Existing output; refuse overwrite/resume: '+str(path))

def sources():
    files=sorted(c.HERE.glob('*.py'))+[c.HERE/'README.md',
      c.ROOT/'scripts/run_ticket_policy_retention_build_v1.py',
      c.ROOT/'scripts/run_ticket_policy_retention_cpu_audit_v1.py',
      c.ROOT/'scripts/package_ticket_policy_retention_candidate_v1.py',
      c.ROOT/'tests/test_ticket_policy_retention_build_v1.py']
    return {f.relative_to(c.ROOT).as_posix():c.sha(f) for f in files}

def dependencies():
    files=['extensions/ticket_policy_retention_pilot_v1/core.py','extensions/ticket_policy_retention_pilot_v1/validate.py',
      'extensions/ticket_sft_v3_stage2/core.py','extensions/ticket_agent_v1/ticket_env.py',
      'extensions/ticket_policy_audit_v1/core.py','extensions/ticket_policy_audit_v1/run.py',
      'extensions/ticket_training_v1_1/support.py','scripts/freeze_ticket_policy_retention_design_v1.py',
      'reports/ticket-policy-retention-design-v1/freeze.json',
      'reports/ticket-policy-retention-design-v1/evaluation-spec.json',
      'reports/ticket-policy-retention-design-v1/evaluation-cases.json',
      'reports/ticket-policy-retention-design-v1/batch-proposal.json',
      'data/ticket_sft_v3_stage2_train_release/token-audit.json']
    return {name:c.sha(c.ROOT/name) for name in files}

def baseline_gate():
    from extensions.ticket_policy_retention_extension_v1_1.pipeline import verify
    result=verify()
    c.require(result['manifest_sha256']==EXT_MANIFEST and result['episodes_sha256']==EXT_EPISODES,'Wrong representative revision')
    review=c.read(c.ROOT/'reports/ticket-policy-retention-extension-v1.1/ai-review.json')
    c.require(review['review_type']=='AI_assisted' and review['blocking_findings']==0 and review['notes_sha256']=='67fd76aee79cc357f39493ca6abdae3514b67ec1770df675ca4c8f6038b77f0b','Wrong representative AI review')
    return result

def accept():
    fresh(c.REPORT)
    result=baseline_gate()
    c.REPORT.mkdir(parents=True)
    old=c.ROOT/'reports/ticket-policy-retention-extension-v1.1'
    receipt={'version':c.VERSION,'recorded_at':now(),'user_text':'确认',
      'context':'Explicit confirmation after corrected 12-pair AI-assisted review and request to enter the bulk stage',
      'accepted_manifest_sha256':EXT_MANIFEST,'accepted_episodes_sha256':EXT_EPISODES,
      'accepted_AI_review_sha256':c.sha(old/'ai-review.json'),'accepted_AI_notes_sha256':c.sha(old/'ai-review-notes.json'),
      'design_freeze_sha256':c.FREEZE_SHA,'evaluation_task_instantiation_authorized':True,
      'bulk_generation_authorized':True,'remote_CPU_audit_authorized':True,
      'training_authorized':False,'gpu_inference_authorized':False,'per_row_human_inspection_attested':False,
      'provenance':'Current-chat user acceptance of AI-assisted review, not independent identity/signature or row-by-row human inspection',
      'legacy_reports_modified':False}
    write(c.REPORT/'user-acceptance.json',receipt)
    write(c.REPORT/'predecessor-verification.json',result)
    return receipt

def acceptance_gate():
    a=c.read(c.REPORT/'user-acceptance.json')
    c.require(a['version']==c.VERSION and a['accepted_manifest_sha256']==EXT_MANIFEST and a['accepted_episodes_sha256']==EXT_EPISODES and a['design_freeze_sha256']==c.FREEZE_SHA,'Acceptance source differs')
    for k in ('evaluation_task_instantiation_authorized','bulk_generation_authorized','remote_CPU_audit_authorized'): c.require(a[k] is True,'Missing stage authorization')
    for k in ('training_authorized','gpu_inference_authorized','per_row_human_inspection_attested'): c.require(a[k] is False,'Authorization overstated')
    old=c.ROOT/'reports/ticket-policy-retention-extension-v1.1'
    c.require(a['accepted_AI_review_sha256']==c.sha(old/'ai-review.json') and a['accepted_AI_notes_sha256']==c.sha(old/'ai-review-notes.json'),'Accepted review changed')
    return a

def run_tests():
    fresh(c.REPORT/'tests.log')
    command=[sys.executable,'-B','-m','unittest','tests.test_ticket_policy_retention_build_v1','-v']
    result=subprocess.run(command,cwd=c.ROOT,capture_output=True,text=True,encoding='utf-8',timeout=600)
    text(c.REPORT/'tests.log',result.stdout+result.stderr)
    c.require(result.returncode==0,'Tests failed; retain captured log')
    return c.sha(c.REPORT/'tests.log')

def oracle(e):
    s=e['spec']; names,errors=c.expected_chain(s)
    return {'id':e['id'],'pair_id':s['pair_id'],'locale':e['locale'],'category':s['theme'],'variant':s['variant'],
      'initial_input':{'system':e['row']['system'],'tools':e['row']['tools'],'messages':[e['row']['conversations'][0]]},
      'fixed_user_supplement':c.supplement(s,e['locale']) if s['followup'] else None,
      'followup_schedule':'After first plain assistant reply regardless of correctness; prior violations persist; total 8 generation/12 tool budgets',
      'fixture':s['fixture'],'fault_schedule':s['fault_schedule'],'public_policy':s['policy'],
      'goals':{'expected_tool_names':names,'expected_error_codes':errors,
               'expected_final_state':e['final_state'],'expected_claims':e['claims'],
               'no_initial_tool_proposals':bool(s['missing_field']) or not names,
               'max_conflict_retries_after_initial':2,'all_requested_bodies_required':s['theme']=='knowledge_chain' and s['variant']!='titles_only'},
      'gold_episode_sha256':c.digest(e),'semantic_review':'required_separately_after_model_inference',
      'model_execution_authorized':False,'development_targeted_not_final_test':True}

def review(episodes,title):
    lines=['# '+title,'','完整模拟 episode；不是模型成绩、人工批准或训练发布。','']
    cards=[]
    for zh,en in zip(episodes[::2],episodes[1::2]):
        s=zh['spec']; label=f'{s["pair_id"]} · {s["theme"]}/{s["variant"]} · {s["permission_profile"]} · allow_write={s["allow_write"]}'
        lines+=['## '+label,'']; card=['<h2>'+html.escape(label)+'</h2>']
        for e in (zh,en):
            chain=' → '.join(x['call']['name']+(' ✓' if x['response']['ok'] else ' ✗ '+x['response']['error']['code']) for x in e['events']) or '无调用 / no calls'
            visible='\n'.join(m['from']+': '+m['value'] for m in e['row']['conversations'] if m['from'] in ('human','gpt'))
            lines+=['### '+e['locale'],'','调用链：'+chain,'','```text',visible,'```','']
            card+=['<h3>'+e['locale']+'</h3><p>'+html.escape(chain)+'</p><pre>'+html.escape(visible)+'</pre>']
        states={'initial':zh['initial_state'],'final':zh['final_state'],'external':zh['external_events']}
        card+=['<details><summary>状态变化</summary><pre>'+html.escape(json.dumps(states,ensure_ascii=False,indent=2))+'</pre></details>',
               '<details><summary>完整中英 JSON（私有元数据不进训练）</summary><pre>'+html.escape(json.dumps([zh,en],ensure_ascii=False,indent=2))+'</pre></details>']
        cards.append('<section>'+''.join(card)+'</section>')
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}section{border-top:1px solid #aaa;padding:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:14px}</style><h1>'+html.escape(title)+'</h1><p>模拟轨迹；AI 辅助审核不等于逐条人工批准。</p>'+''.join(cards)+'</html>'
    return '\n'.join(lines)+'\n',page

def freeze_eval():
    acceptance_gate(); baseline_gate(); fresh(c.EVAL)
    test_sha=run_tests()
    c.EVAL.mkdir(parents=True)
    write(c.EVAL/'started.json',{'stage':'targeted_eval_freeze','started_at':now(),'training_started':False})
    try:
        episodes=[c.generate(s,l) for s in c.specs('eval') for l in ('zh','en')]
        write(c.EVAL/'gold-episodes.json',episodes)
        write(c.EVAL/'tasks.json',[oracle(e) for e in episodes])
        result=v.population(episodes,'eval',v.blocked_sources()); write(c.EVAL/'validation.json',result)
        manifest={'version':c.VERSION,'scope':'Frozen 192 targeted development evaluations, not final test or inference results',
          'frozen_at':now(),'rows':192,'pairs':96,'seed':c.SEEDS['eval'],'source_sha256':sources(),'dependencies_sha256':dependencies(),
          'design_freeze_sha256':c.FREEZE_SHA,'acceptance_sha256':c.sha(c.REPORT/'user-acceptance.json'),'tests_sha256':test_sha,
          'files_sha256':{n:c.sha(c.EVAL/n) for n in ('gold-episodes.json','tasks.json','validation.json')},
          'runtime':c.read(c.DESIGN/'evaluation-spec.json')['runtime_proposal'],
          'bulk_not_yet_built':not c.OUT.exists(),'model_inference_started':False,'training_started':False,'old_final_test_read':False}
        c.require(manifest['bulk_not_yet_built'],'Targeted eval must freeze before bulk')
        write(c.EVAL/'manifest.json',manifest)
        write(c.EVAL/'status.json',{'status':'complete','manifest_sha256':c.sha(c.EVAL/'manifest.json'),'completed_at':now()})
        return {k:x for k,x in result.items() if k!='records'}|{'manifest_sha256':c.sha(c.EVAL/'manifest.json')}
    except BaseException as exc:
        text(c.EVAL/'failure.log',traceback.format_exc()); write(c.EVAL/'failure.json',{'error':repr(exc),'failed_at':now()}); raise

def eval_gate(replay=True):
    acceptance_gate(); m=c.read(c.EVAL/'manifest.json'); status=c.read(c.EVAL/'status.json')
    c.require(m['rows']==192 and m['pairs']==96 and m['source_sha256']==sources() and m['dependencies_sha256']==dependencies(),'Evaluation source/count changed')
    c.require(m['design_freeze_sha256']==c.FREEZE_SHA and m['acceptance_sha256']==c.sha(c.REPORT/'user-acceptance.json') and m['tests_sha256']==c.sha(c.REPORT/'tests.log'),'Evaluation authority/test changed')
    c.require(m['bulk_not_yet_built'] and not m['model_inference_started'] and not m['training_started'] and status['status']=='complete' and status['manifest_sha256']==c.sha(c.EVAL/'manifest.json'),'Evaluation freeze incomplete')
    c.require(set(m['files_sha256'])=={'gold-episodes.json','tasks.json','validation.json'},'Incomplete eval binding')
    for n,h in m['files_sha256'].items(): c.require(c.sha(c.EVAL/n)==h,'Frozen eval artifact changed: '+n)
    episodes=c.read(c.EVAL/'gold-episodes.json')
    c.require(c.read(c.EVAL/'tasks.json')==[oracle(e) for e in episodes],'Frozen input/oracle differs')
    if replay: c.require(v.population(episodes,'eval',v.blocked_sources())==c.read(c.EVAL/'validation.json'),'Frozen oracle replay differs')
    return m,episodes

def build():
    acceptance_gate(); baseline_gate(); eval_manifest,_=eval_gate(); fresh(c.OUT)
    c.OUT.mkdir(parents=True); write(c.OUT/'started.json',{'stage':'candidate_build','started_at':now(),
      'targeted_eval_manifest_sha256':c.sha(c.EVAL/'manifest.json'),'training_started':False})
    try:
        # Only spec/quota reservations go into generation, no evaluation gold.
        episodes=[c.generate(s,l) for s in c.specs('train') for l in ('zh','en')]
        write(c.OUT/'episodes.json',episodes)
        blocked=v.blocked_sources(); validation=v.population(episodes,'train',blocked)
        write(c.OUT/'validation.json',validation)
        # Deliberately separate POSTBUILD evaluator; it never designs/generates rows.
        evaluation=c.read(c.EVAL/'gold-episodes.json')
        write(c.OUT/'isolation.json',v.isolation(episodes,evaluation,blocked))
        write(c.OUT/'train.json',[e['row'] for e in episodes]); write(c.OUT/'dataset_info.json',c.dataset_info())
        write(c.OUT/'supervision.json',SUPERVISION)
        write(c.OUT/'coverage.json',coverage(episodes))
        names=('episodes.json','validation.json','isolation.json','train.json','dataset_info.json','supervision.json','coverage.json')
        manifest={'version':c.VERSION,'built_at':now(),'rows':1800,'pairs':900,'seed':c.SEEDS['train'],
          'sources_sha256':sources(),'dependencies_sha256':dependencies(),'files_sha256':{n:c.sha(c.OUT/n) for n in names},
          'targeted_eval_manifest_sha256':c.sha(c.EVAL/'manifest.json'),'targeted_eval_frozen_at':eval_manifest['frozen_at'],
          'acceptance_sha256':c.sha(c.REPORT/'user-acceptance.json'),'tests_sha256':c.sha(c.REPORT/'tests.log'),
          'supervision':SUPERVISION,'actual_token_label_audit':'pending_remote','sample_review':'pending_post_CPU_audit',
          'ready_for_training':False,'training_started':False,'gpu_inference_started':False,'old_final_test_read':False}
        write(c.OUT/'manifest.json',manifest)
        write(c.OUT/'status.json',{'status':'complete','manifest_sha256':c.sha(c.OUT/'manifest.json'),'completed_at':now()})
        return summary(validation)|{'candidate_manifest_sha256':c.sha(c.OUT/'manifest.json'),'targeted_eval_manifest_sha256':c.sha(c.EVAL/'manifest.json')}
    except BaseException as exc:
        text(c.OUT/'failure.log',traceback.format_exc()); write(c.OUT/'failure.json',{'error':repr(exc),'failed_at':now(),'failed_rows_dropped':False}); raise

def coverage(episodes):
    from collections import Counter
    specs=[e['spec'] for e in episodes[::2]]
    return {'theme_variant_profile_pairs':dict(Counter('|'.join(s[k] for k in ('theme','variant','permission_profile')) for s in specs)),
      'theme_permission_locale_rows':dict(Counter(f'{e["spec"]["theme"]}|{e["spec"]["allow_write"]}|{e["locale"]}' for e in episodes)),
      'missing_field_pairs':dict(Counter(s['missing_field'] or 'none' for s in specs)),
      'followup_pairs':sum(s['followup'] for s in specs),'external_fault_count_pairs':dict(Counter(str(len(s['fault_schedule'])) for s in specs)),
      'business_families_pairs':dict(Counter(s['business_family'] for s in specs)),
      'surface_families_pairs':dict(Counter(s['surface_family'] for s in specs)),
      'source_bodies':len({a['content'] for s in specs for a in s['fixture']['articles']}),
      'first_ticket_states_pairs':dict(Counter(s['fixture']['tickets'][0]['status'] for s in specs)),
      'initial_versions':{'min':min(x['version'] for s in specs for x in s['fixture']['tickets']), 'max':max(x['version'] for s in specs for x in s['fixture']['tickets'])}}

def summary(result): return {k:x for k,x in result.items() if k!='records'}

def verify():
    acceptance_gate(); baseline_gate(); eval_gate()
    m=c.read(c.OUT/'manifest.json'); status=c.read(c.OUT/'status.json')
    c.require(m['rows']==1800 and m['pairs']==900 and m['seed']==c.SEEDS['train'] and m['sources_sha256']==sources() and m['dependencies_sha256']==dependencies(),'Candidate source/count changed')
    c.require(m['targeted_eval_manifest_sha256']==c.sha(c.EVAL/'manifest.json') and m['acceptance_sha256']==c.sha(c.REPORT/'user-acceptance.json') and m['tests_sha256']==c.sha(c.REPORT/'tests.log'),'Candidate source evidence changed')
    c.require(m['supervision']==SUPERVISION and c.read(c.OUT/'supervision.json')==SUPERVISION,'Supervision changed')
    for k in ('ready_for_training','training_started','gpu_inference_started','old_final_test_read'): c.require(m[k] is False,'Scope overstated')
    c.require(m['actual_token_label_audit']=='pending_remote' and m['sample_review']=='pending_post_CPU_audit','Historical candidate stage changed')
    c.require(status['status']=='complete' and status['manifest_sha256']==c.sha(c.OUT/'manifest.json'),'Build incomplete')
    names={'episodes.json','validation.json','isolation.json','train.json','dataset_info.json','supervision.json','coverage.json'}
    c.require(set(m['files_sha256'])==names,'Incomplete candidate bindings')
    for n,h in m['files_sha256'].items(): c.require(c.sha(c.OUT/n)==h,'Candidate artifact changed: '+n)
    episodes=c.read(c.OUT/'episodes.json'); blocked=v.blocked_sources()
    validation=v.population(episodes,'train',blocked)
    c.require(validation==c.read(c.OUT/'validation.json'),'Full replay validation differs')
    c.require(v.isolation(episodes,c.read(c.EVAL/'gold-episodes.json'),blocked)==c.read(c.OUT/'isolation.json'),'Isolation differs')
    c.require([e['row'] for e in episodes]==c.read(c.OUT/'train.json') and c.dataset_info()==c.read(c.OUT/'dataset_info.json') and coverage(episodes)==c.read(c.OUT/'coverage.json'),'Export/coverage differs')
    return summary(validation)|{'candidate_manifest_sha256':c.sha(c.OUT/'manifest.json'),'targeted_eval_manifest_sha256':c.sha(c.EVAL/'manifest.json')}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',required=True,choices=('accept','freeze-eval','build','verify'))
    a=parser.parse_args(); result={'accept':accept,'freeze-eval':freeze_eval,'build':build,'verify':verify}[a.stage]()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__': main()
