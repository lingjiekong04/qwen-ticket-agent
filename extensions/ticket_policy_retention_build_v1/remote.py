"""Pinned real tokenizer/converter/preprocessor CPU audit; no model execution."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
import traceback
from . import core as c,pipeline as p
from extensions.ticket_policy_audit_v1 import core as audit

SAMPLE={'knowledge_chain':16,'authority':12,'clarification':8,'faithful':8,'priority':8,'retention':8}

def features(e):
    s=e['spec']; v=s['variant']
    keys={'variant:'+v,'profile:'+s['permission_profile']}
    if s['source_language'] and s['theme']=='knowledge_chain': keys.add('source:'+s['source_language'])
    if v=='conflict_success': keys.add('conflict_count:'+str(len(s['fault_schedule'])))
    if v=='read_two': keys.add('dual_read_equal:'+str(len({x['priority'] for x in s['fixture']['tickets']})==1))
    return keys

def select_sample(episodes,lengths):
    c.require(set(lengths)=={e['id'] for e in episodes} and all(isinstance(x,int) and 0<x<=4096 for x in lengths.values()),'Sample requires all real token lengths')
    pairs=episodes[::2]; chosen=[]; evidence={}
    rng=random.Random(c.SEEDS['train']); ranks={e['id']:rng.random() for e in pairs}
    for theme,budget in SAMPLE.items():
        candidates=[e for e in pairs if e['spec']['theme']==theme]
        units=[]
        for e in candidates:
            if theme=='authority':
                group=e['spec']['contrast_group']
                if any(u[0]['spec']['contrast_group']==group for u in units): continue
                unit=[x for x in candidates if x['spec']['contrast_group']==group]
                c.require(len(unit)==2,'Counterfactual sampling group incomplete')
            else: unit=[e]
            units.append(unit)
        wanted=set().union(*(features(e) for e in candidates))
        longest=max(candidates,key=lambda e:(max(lengths[e['id']],lengths[e['spec']['pair_id']+'-en']),e['id']))
        selected=next(u for u in units if longest in u).copy(); covered=set().union(*(features(e) for e in selected))
        remaining=[u for u in units if not any(e in selected for e in u)]
        while len(selected)<budget:
            fitting=[u for u in remaining if len(u)+len(selected)<=budget]
            c.require(fitting,'Cannot fill fixed sample quota')
            unit=max(fitting,key=lambda u:(len(set().union(*(features(e) for e in u))-covered), max(lengths[e['id']] for e in u), ranks[u[0]['id']]))
            selected+=unit; covered|=set().union(*(features(e) for e in unit)); remaining.remove(unit)
        c.require(wanted<=covered,'Fixed sample misses required variant/profile/source/risk: '+str(wanted-covered))
        evidence[theme]={'pairs':budget,'longest_pair_id':longest['spec']['pair_id'],'covered_features':sorted(covered),
                         'sampling_unit':'complete permission counterfactual group' if theme=='authority' else 'bilingual pair'}
        chosen+=selected
    ids={e['spec']['pair_id'] for e in chosen}
    rows=[e for e in episodes if e['spec']['pair_id'] in ids]
    c.require(len(ids)==60 and len(rows)==120,'Wrong review sample size')
    return rows,{'seed':c.SEEDS['train'],'pairs':60,'rows':120,'allocation':SAMPLE,'strata':evidence,
      'method':'Category longest pair first, deterministic feature-cover selection with seeded tie break, fill fixed quotas; all variants, profiles, body languages and listed recovery/dual-read risk controls',
      'not_all_cross_product_cells_sampled':True,'all_population_cross_product_cells_automatically_validated':True,
      'review_type':'pending_AI_assisted_then_explicit_user_acceptance','per_row_human_inspection_attested':False}

def audit_remote(directory,model_path):
    audit.cpu_gate(os.environ)
    verified=p.verify()
    old=c.read(audit.s.RELEASE/'token-audit.json')
    from extensions.ticket_policy_audit_v1.run import load_cpu_components
    versions,tokenizer,template,_,processor=load_cpu_components(model_path,old)
    from llamafactory.data.converter import SharegptDatasetConverter
    from llamafactory.data.parser import DatasetAttr
    from llamafactory.hparams import DataArguments
    attr=DatasetAttr('file','train.json'); attr.join(c.dataset_info()['ticket_policy_retention_candidate_v1'])
    data_args=DataArguments(**p.SUPERVISION)
    converter=SharegptDatasetConverter(attr,data_args)
    episodes=c.read(c.OUT/'episodes.json'); records=[]
    with (directory/'records.jsonl').open('x',encoding='utf-8') as stream:
        for i,e in enumerate(episodes):
            row=e['row']; converted=converter(row)
            messages=audit.check_converted(row,converted)
            pairs=template.encode_multiturn(tokenizer,messages,row['system'],row['tools'])
            actual=processor.preprocess_dataset({k:[v] for k,v in converted.items()})
            c.require(len(actual['input_ids'])==len(actual['labels'])==1,'Dropped row: '+e['id'])
            ids,labels=actual['input_ids'][0],actual['labels'][0]
            turns=sum(m['from'] in ('gpt','function_call') for m in row['conversations'])
            metrics=audit.verify_labels(pairs,ids,labels,turns)
            permission=row['system'].split('\n')[-1]
            span=audit.text_span(tokenizer,list(pairs[0][0]),permission)
            c.require(span['end']<=len(pairs[0][0]) and all(v==-100 for v in labels[span['start']:span['end']]),'Permission missing or wrongly supervised')
            record={'id':e['id'],'row_sha256':c.digest(row),**metrics,'permission_span':span,
              'permission_labels_masked_as_expected':True,'all_assistant_turns_supervised':True,
              'public_source_spans_masked':True,'converter_messages_sha256':c.digest(messages)}
            records.append(record); stream.write(c.canonical(record)+'\n'); stream.flush()
            if (i+1)%100==0: print(f'Actual CPU tokenizer/labels: {i+1}/1800',flush=True)
    sample,selection=select_sample(episodes,{r['id']:r['total_tokens'] for r in records})
    p.write(directory/'review-episodes.json',sample); p.write(directory/'review-pairs.json',selection)
    md,page=p.review(sample,'固定分层抽检：60 对／120 条')
    p.text(directory/'review.md',md); p.text(directory/'review.html',page)
    import torch
    c.require(not torch.cuda.is_initialized(),'CUDA unexpectedly initialized')
    files=('records.jsonl','review-episodes.json','review-pairs.json','review.md','review.html')
    return {'passed':True,'rows':1800,'pairs':900,'candidate_manifest_sha256':verified['candidate_manifest_sha256'],
      'targeted_eval_manifest_sha256':verified['targeted_eval_manifest_sha256'],'sources_sha256':p.sources(),
      'dependencies_sha256':p.dependencies(),'versions':versions,'python':sys.version.split()[0],
      'tokenizer_asset_sha256':old['tokenizer_asset_sha256'],'framework_source_sha256':old['framework_source_sha256'],
      'supervision':p.SUPERVISION,'ordered_records_sha256':c.digest(records),
      'files_sha256':{name:c.sha(directory/name) for name in files},
      'max_tokens':max(r['total_tokens'] for r in records),'assistant_turns':sum(r['assistant_turns'] for r in records),
      'all_rows_untruncated':True,'all_assistant_turns_supervised':True,'real_tokenizer_audited':True,
      'sample_pairs':60,'sample_rows':120,'sample_review':'pending','ready_for_training':False,
      'training_started':False,'model_weights_loaded':False,'cuda_initialized':False,'old_final_test_read':False}

def compare(directory):
    audit.cpu_gate(os.environ); verified=p.verify(); remote=c.REPORT/'remote'
    report=c.read(remote/'report.json'); status=c.read(remote/'status.json')
    c.require(status['status']=='complete' and status['report_sha256']==c.sha(remote/'report.json') and report['passed'] and report['rows']==1800 and report['pairs']==900,'Remote audit incomplete')
    c.require(report['candidate_manifest_sha256']==verified['candidate_manifest_sha256'] and report['targeted_eval_manifest_sha256']==verified['targeted_eval_manifest_sha256'] and report['sources_sha256']==p.sources() and report['dependencies_sha256']==p.dependencies(),'Remote source binding differs')
    c.require(report['versions']==audit.s.STACK and report['python'].startswith('3.11.') and report['supervision']==p.SUPERVISION,'Pinned runtime or supervision changed')
    old=c.read(audit.s.RELEASE/'token-audit.json')
    for key in ('tokenizer_asset_sha256','framework_source_sha256'): c.require(report[key]==old[key],'Pinned tokenizer/framework evidence changed')
    for key in ('all_rows_untruncated','all_assistant_turns_supervised','real_tokenizer_audited'): c.require(report[key] is True,'Missing audit check')
    for key in ('ready_for_training','training_started','model_weights_loaded','cuda_initialized','old_final_test_read'): c.require(report[key] is False,'Scope changed')
    c.require(set(report['files_sha256'])=={'records.jsonl','review-episodes.json','review-pairs.json','review.md','review.html'},'Evidence binding incomplete')
    for name,h in report['files_sha256'].items(): c.require(Path(name).name==name and c.sha(remote/name)==h,'Remote evidence changed')
    records=[json.loads(line) for line in (remote/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    episodes=c.read(c.OUT/'episodes.json')
    c.require(len(records)==len(episodes)==1800 and c.digest(records)==report['ordered_records_sha256'],'Dropped/reordered record')
    for e,r in zip(episodes,records):
        c.require(r['id']==e['id'] and r['row_sha256']==c.digest(e['row']) and r['passed'] and 0<r['total_tokens']<=4096,'Unbound record')
        turns=sum(m['from'] in ('gpt','function_call') for m in e['row']['conversations'])
        expected=[{'role':audit.ROLES[m['from']],'content':m['value']} for m in e['row']['conversations']]
        c.require(r['converter_messages_sha256']==c.digest(expected) and r['assistant_turns']==turns==len(r['target_spans']),'Converted roles or supervision turns differ')
        spans=r['target_spans']; last=0
        for s in spans:
            c.require(last<s['start']<s['end']<=r['total_tokens'] and s['target_tokens']==s['end']-s['start'],'Invalid supervised span'); last=s['end']
        c.require(last==r['total_tokens'] and sum(s['target_tokens'] for s in spans)==r['supervised_tokens'],'Incomplete assistant labels')
        q=r['permission_span']
        c.require(q['text']==e['row']['system'].split('\n')[-1] and q['source_visible'] and 0<=q['start']<q['end']<=spans[0]['start'] and q['end']-q['start']==len(q['token_ids']) and q['token_ids_sha256']==c.digest(q['token_ids']),'Permission source span changed')
        for key in ('permission_labels_masked_as_expected','all_assistant_turns_supervised','public_source_spans_masked'): c.require(r[key] is True,'Incorrect mask claim')
        for key in ('input_ids_sha256','labels_sha256'): c.require(len(r[key])==64 and all(ch in '0123456789abcdef' for ch in r[key]),'Missing token/label hash')
    c.require(report['max_tokens']==max(r['total_tokens'] for r in records) and report['assistant_turns']==sum(r['assistant_turns'] for r in records),'Aggregate differs')
    sample,selection=select_sample(episodes,{r['id']:r['total_tokens'] for r in records})
    c.require(sample==c.read(remote/'review-episodes.json') and selection==c.read(remote/'review-pairs.json'),'Fixed sample changed')
    md,page=p.review(sample,'固定分层抽检：60 对／120 条')
    c.require((remote/'review.md').read_text(encoding='utf-8')==md and (remote/'review.html').read_text(encoding='utf-8')==page,'Review page changed')
    return {'passed':True,'rows':1800,'pairs':900,'max_tokens':report['max_tokens'],'assistant_turns':report['assistant_turns'],
      'candidate_manifest_sha256':verified['candidate_manifest_sha256'],'targeted_eval_manifest_sha256':verified['targeted_eval_manifest_sha256'],
      'remote_report_sha256':c.sha(remote/'report.json'),'real_tokenizer_audited':True,'sample_pairs':60,'sample_rows':120,
      'comparison_kind':'Hash and consistency verification of server evidence, not independent tokenizer re-encoding',
      'sample_review':'pending_AI_assisted_then_user_acceptance','ready_for_training':False,'training_started':False,'model_weights_loaded':False}

def main():
    a=argparse.ArgumentParser(); a.add_argument('--stage',choices=('remote','compare'),required=True)
    a.add_argument('--model-path',type=Path,default=Path(audit.s.MODEL)); args=a.parse_args()
    directory=c.REPORT/args.stage; p.fresh(directory); directory.mkdir(parents=True)
    p.write(directory/'started.json',{'stage':args.stage,'started_at':p.now(),'training_started':False})
    try:
        result=audit_remote(directory,args.model_path) if args.stage=='remote' else compare(directory)
        p.write(directory/'report.json',result); p.write(directory/'status.json',{'status':'complete','report_sha256':c.sha(directory/'report.json'),'completed_at':p.now()})
        print(json.dumps({k:result[k] for k in ('passed','rows','sample_pairs','sample_review','ready_for_training','training_started')}),flush=True)
    except BaseException as exc:
        p.text(directory/'failure.log',traceback.format_exc()); p.write(directory/'failure.json',{'status':'failed','error':repr(exc),'training_started':False}); raise
if __name__=='__main__': main()
