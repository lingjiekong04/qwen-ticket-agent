"""Build/verify candidate; separately run CPU remote audit. No training."""
import argparse
from collections import Counter,defaultdict
import copy
import json
from pathlib import Path
import random
from . import core as c
from .validate import forbidden,normalized,validate_population
s=c.s


def gates():
    root=c.ROOT/'reports/ticket-policy-repair-next-stage'
    record=s.read(root/'verification.json')
    for name,sha in record['files_sha256'].items():
        s.require(Path(name).name==name and s.sha(root/name)==sha,'Accepted plan evidence changed')
    from scripts.record_ticket_policy_repair_confirmation import validate_plan
    validate_plan(s.read(root/'plan.json'))
    accept=s.read(root/'user-design-confirmation.json')
    s.require(accept['design_confirmation']=='accepted' and accept['rows']==36 and accept['pairs']==18,'Representative design not accepted')
    for path,sha in accept['bound_files_sha256'].items(): s.require(s.sha(c.ROOT/path)==sha,'Pilot evidence changed')
    s.require(s.sha(c.ROOT/'lora_sft_ticket_policy_repair_pilot_v1.zip')==accept['pilot_zip_sha256'],'Pilot ZIP changed')
    c.audit.delivery_gate()
    c.audit.base_gate()
    return {'plan_sha256':s.sha(root/'plan.json'),'plan_document_sha256':s.sha(root/'plan.md'),
        'design_confirmation_sha256':s.sha(root/'user-design-confirmation.json'),
        'pilot_sources':c.base.source_hashes(),'frozen_dev_sha256':s.DEV_SHA,'release_manifest_sha256':s.RELEASE_SHA}


def dataset_info():
    return {'ticket_policy_repair_candidate_v1':{'file_name':'train.json','formatting':'sharegpt',
        'columns':{'messages':'conversations','system':'system','tools':'tools'},
        'tags':{'role_tag':'from','content_tag':'value','user_tag':'human','assistant_tag':'gpt',
                'function_tag':'function_call','observation_tag':'observation'}}}


def select_sample(episodes,lengths=None):
    pairs=[e for e in episodes if e['locale']=='zh']
    score=lambda e: max(lengths[e['spec']['pair_id']+'-'+loc] for loc in ('zh','en')) if lengths else len(c.audit.canonical(e['row']))
    selected=[]
    # Two entire permission groups of each business variant, while covering all profiles.
    chosen_profiles=set()
    for stem in ('create','override','assign'):
        candidates=[e for e in pairs if e['spec']['variant']==stem+'_denied']
        candidates.sort(key=lambda e:(-score(e),s.digest([c.SEED,e['id']])))
        first=candidates[0]
        extra=next((e for e in candidates[1:] if e['spec']['profile'] not in chosen_profiles|{first['spec']['profile']}),candidates[1])
        for e in (first,extra):
            chosen_profiles.add(e['spec']['profile'])
            selected.extend(x['spec']['pair_id'] for x in pairs if x['spec']['contrast_group']==e['spec']['contrast_group'])
    s.require(len(chosen_profiles)==3,'Authority sample missed profile')
    for category in ('clarify','faithful','priority'):
        candidates=[e for e in pairs if e['spec']['category']==category]
        candidates.sort(key=lambda e:(-score(e),s.digest([c.SEED,e['id']])))
        chosen=[candidates[0]]
        for variant in c.VARIANTS[category]:
            if not any(e['spec']['variant']==variant for e in chosen): chosen.append(next(e for e in candidates if e['spec']['variant']==variant))
        for profile in ('json','sentence','explicit'):
            if not any(e['spec']['profile']==profile for e in chosen): chosen.append(next(e for e in candidates if e['spec']['profile']==profile))
        rng=random.Random(f'{c.SEED}:review:{category}')
        rest=[e for e in candidates if e not in chosen]
        rng.shuffle(rest)
        chosen+=rest[:12-len(chosen)]
        selected.extend(e['spec']['pair_id'] for e in chosen)
    s.require(len(selected)==len(set(selected))==48,'Sample count differs')
    sampled=[e for e in episodes if e['spec']['pair_id'] in selected]
    s.require(Counter(e['spec']['category'] for e in sampled)=={k:24 for k in c.QUOTAS},'Sample strata differ')
    return sampled


def render_review(sample,title):
    from extensions.ticket_policy_repair_pilot_v1.pipeline import review
    md,page=review(sample)
    md=md.replace('修复代表样例：18 对 / 36 条',title).replace('3 组权限对照','6 组权限对照')
    page=page.replace('修复代表样例：18 对 / 36 条',title)
    return md,page


def similarity_report(episodes,blocked):
    # Conservative lexical tri-gram screening, not a semantic leakage oracle.
    def grams(text):
        value=normalized(text)
        return {value[i:i+3] for i in range(max(0,len(value)-2))}
    old=[(s.digest(t),grams(t)) for t in sorted(blocked['requests'])]
    inverted=defaultdict(set)
    for i,(_,gs) in enumerate(old):
        for gram in gs: inverted[gram].add(i)
    flagged=[]
    for e in episodes:
        query=grams(e['row']['conversations'][0]['value'])
        counts=Counter(i for gram in query for i in inverted.get(gram,()))
        if not counts: continue
        index,overlap=counts.most_common(1)[0]
        similarity=overlap/len(query|old[index][1])
        if similarity>=0.8: flagged.append({'id':e['id'],'prior_request_sha256':old[index][0],'trigram_jaccard':similarity})
    return {'method':'Normalized character trigram Jaccard; exact normalized overlaps rejected separately',
        'threshold':0.8,'flags':flagged,'flags_human_disposition':'pending' if flagged else 'not_applicable',
        'shared_schema_and_tool_names_are_not_alone_leakage':True}


def build():
    binding=gates()
    s.fresh(c.OUT)
    s.fresh(c.REPORT)
    c.OUT.mkdir(parents=True)
    c.REPORT.mkdir(parents=True)
    s.write(c.OUT/'status.json',{'status':'started','training_started':False})
    try:
        s.write(c.REPORT/'batch-authorization.json',{'user_message':'开始','scope':'Build 1200 repair candidate episodes only; no training',
            'provenance':'Explicit current-chat user authorization, not a signature','binding':binding,
            'batch_generation_authorized':True,'training_authorized':False})
        episodes=[]
        for spec in c.specs():
            for locale in ('zh','en'): episodes.append(c.generate(spec,locale))
            if len(episodes)%100==0: print(f'Executed candidate episodes: {len(episodes)}/1200',flush=True)
        s.write(c.OUT/'episodes.json',episodes)
        blocked=forbidden()
        validation=validate_population(episodes,blocked,progress=True)
        s.write(c.OUT/'validation.json',validation)
        similarity=similarity_report(episodes,blocked)
        s.write(c.OUT/'isolation-similarity.json',similarity)
        s.write(c.OUT/'train.json',[e['row'] for e in episodes])
        s.write(c.OUT/'dataset_info.json',dataset_info())
        coverage=Counter((e['spec']['category'],e['spec']['variant'],e['spec']['profile'],e['locale'],e['spec']['policy']['allow_write']) for e in episodes)
        s.write(c.OUT/'coverage.json',[{'category':k[0],'variant':k[1],'profile':k[2],'locale':k[3],'allow_write':k[4],'rows':v} for k,v in sorted(coverage.items())])
        sample=select_sample(episodes)
        s.write(c.OUT/'provisional-review-pairs.json',[e['spec']['pair_id'] for e in sample if e['locale']=='zh'])
        md,page=render_review(sample,'修复候选预览：48 对 / 96 条（字符长度抽样；真实 token 审计后重选）')
        (c.OUT/'review.md').write_text(md,encoding='utf-8')
        (c.OUT/'review.html').write_text(page,encoding='utf-8')
        files=[p for p in c.OUT.iterdir() if p.is_file() and p.name!='status.json']
        s.write(c.OUT/'manifest.json',{'revision':c.REVISION,'seed':c.SEED,'rows':1200,'pairs':600,'quotas':c.QUOTAS,
            'sources':c.source_hashes(),'binding':binding,'authorization_sha256':s.sha(c.REPORT/'batch-authorization.json'),
            'files_sha256':{p.name:s.sha(p) for p in files},'ready_for_training':False,'human_review':'pending',
            'real_tokenizer_audited':False,'training_started':False,'model_weights_loaded':False,'old_final_test_read':False,
            'isolation':{k:blocked[k] for k in ('frozen_dev_sha256','old_train_sha256','diagnostic_tasks_sha256')}})
        s.update(c.OUT/'status.json',{'status':'complete','manifest_sha256':s.sha(c.OUT/'manifest.json'),'training_started':False})
        print(json.dumps({'passed':True,'rows':1200,'pairs':600,'token_audit':'pending_remote','human_review':'pending','training_started':False}))
    except BaseException as exc:
        s.update(c.OUT/'status.json',{'status':'failed','error':repr(exc),'training_started':False})
        raise


def verify(replay=True):
    manifest=s.read(c.OUT/'manifest.json')
    status=s.read(c.OUT/'status.json')
    s.require(status['status']=='complete' and status['manifest_sha256']==s.sha(c.OUT/'manifest.json'),'Incomplete/changed candidate')
    s.require(manifest['sources']==c.source_hashes() and manifest['binding']==gates(),'Source binding changed')
    s.require(s.sha(c.REPORT/'batch-authorization.json')==manifest['authorization_sha256'],'Authorization record changed')
    s.require(not manifest['ready_for_training'] and not manifest['training_started'] and not manifest['real_tokenizer_audited'],'Premature release or training')
    for name,sha in manifest['files_sha256'].items(): s.require(Path(name).name==name and s.sha(c.OUT/name)==sha,'Candidate bytes changed: '+name)
    episodes=s.read(c.OUT/'episodes.json')
    s.require(s.read(c.OUT/'train.json')==[e['row'] for e in episodes] and s.read(c.OUT/'dataset_info.json')==dataset_info(),'Training projection changed')
    if replay: s.require(validate_population(episodes,forbidden())==s.read(c.OUT/'validation.json'),'Replayed validation differs')
    return manifest,episodes


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('build','verify'))
    args=parser.parse_args()
    if args.command=='build': build()
    else:
        verify()
        print(json.dumps({'passed':True,'rows':1200,'ready_for_training':False,'training_started':False}))
