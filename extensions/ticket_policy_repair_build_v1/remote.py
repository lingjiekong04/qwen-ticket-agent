"""Pinned, offline CPU-only tokenizer audit and evidence comparison."""
import argparse
from collections import defaultdict
import inspect
import json
from pathlib import Path
import os
from . import core as c
from .pipeline import verify,select_sample,render_review,dataset_info
s=c.s


def permission_text(spec):
    if spec['profile']=='json': return c.audit.MARKER+c.audit.canonical(spec['policy'])
    return f"Write actions are {'permitted' if spec['policy']['allow_write'] else 'NOT permitted'} by the trusted client policy."


def audit(directory,model_path):
    c.audit.cpu_gate(os.environ)
    manifest,episodes=verify()
    from extensions.ticket_policy_audit_v1.run import load_cpu_components
    old=s.read(s.RELEASE/'token-audit.json')
    versions,tokenizer,template,_,processor=load_cpu_components(model_path,old)
    from llamafactory.data.converter import SharegptDatasetConverter
    from llamafactory.data.parser import DatasetAttr
    from llamafactory.hparams import DataArguments
    attr=DatasetAttr('file','train.json')
    attr.join(dataset_info()['ticket_policy_repair_candidate_v1'])
    data_args=DataArguments(template='qwen',cutoff_len=4096,train_on_prompt=False,mask_history=False,packing=False)
    converter=SharegptDatasetConverter(attr,data_args)
    records=[]
    with (directory/'records.jsonl').open('x',encoding='utf-8') as stream:
        for episode in episodes:
            row=episode['row']
            converted=converter(row)
            messages=c.audit.check_converted(row,converted)
            pairs=template.encode_multiturn(tokenizer,messages,row['system'],row['tools'])
            actual=processor.preprocess_dataset({k:[v] for k,v in converted.items()})
            s.require(len(actual['input_ids'])==len(actual['labels'])==1,'Preprocessor dropped '+episode['id'])
            ids,labels=actual['input_ids'][0],actual['labels'][0]
            assistant=[m for m in row['conversations'] if m['from'] in {'gpt','function_call'}]
            metrics=c.audit.verify_labels(pairs,ids,labels,len(assistant))
            span=c.audit.text_span(tokenizer,list(pairs[0][0]),permission_text(episode['spec']))
            s.require(span['end']<=metrics['target_spans'][0]['start'] and all(x==-100 for x in labels[span['start']:span['end']]),'Permission missing/masked incorrectly')
            calls=sum(len(target) for (_,target),m in zip(pairs,assistant) if m['from']=='function_call')
            record={'id':episode['id'],'pair_id':episode['spec']['pair_id'],'category':episode['spec']['category'],
                'variant':episode['spec']['variant'],'profile':episode['spec']['profile'],'locale':episode['locale'],
                'row_sha256':s.digest(row),**metrics,'call_target_tokens':calls,
                'text_target_tokens':metrics['supervised_tokens']-calls,'permission_span':span,
                'permission_labels_masked_as_expected':True,'all_assistant_turns_supervised':True,
                'public_source_spans_masked':True,'converted_turns':len(messages)}
            records.append(record)
            stream.write(c.audit.canonical(record)+'\n')
            stream.flush()
            if len(records)%100==0: print(f'CPU repair tokenizer/label audit: {len(records)}/1200',flush=True)
    s.require(len(records)==1200,'Audit incomplete')
    lengths={r['id']:r['total_tokens'] for r in records}
    sample=select_sample(episodes,lengths)
    s.write(directory/'review-pairs.json',[e['spec']['pair_id'] for e in sample if e['locale']=='zh'])
    s.write(directory/'review-episodes.json',sample)
    md,page=render_review(sample,'修复候选正式抽检：48 对 / 96 条（真实 token 长度分层）')
    (directory/'review.md').write_text(md,encoding='utf-8')
    (directory/'review.html').write_text(page,encoding='utf-8')
    coverage=defaultdict(lambda:defaultdict(int))
    for r in records:
        key=tuple(r[k] for k in ('category','locale','profile','variant'))
        for field in ('assistant_turns','total_tokens','supervised_tokens','call_target_tokens','text_target_tokens'): coverage[key][field]+=r[field]
        coverage[key]['rows']+=1
    import torch
    s.require(not torch.cuda.is_initialized(),'CUDA initialized during CPU audit')
    return {'passed':True,'rows':1200,'pairs':600,'candidate_manifest_sha256':s.sha(c.OUT/'manifest.json'),
        'sources':c.source_hashes(),'versions':versions,'python_requirement':'3.11',
        'tokenizer_asset_sha256':old['tokenizer_asset_sha256'],'framework_source_sha256':old['framework_source_sha256'],
        'model_path':str(model_path.resolve()),'max_tokens':max(lengths.values()),
        'assistant_turns':sum(r['assistant_turns'] for r in records),
        'coverage':[dict(zip(('category','locale','profile','variant'),k),**v) for k,v in sorted(coverage.items())],
        'records_sha256':s.sha(directory/'records.jsonl'),'ordered_records_sha256':s.digest(records),
        'sample_pairs':48,'sample_rows':96,
        'files_sha256':{p.name:s.sha(p) for p in directory.iterdir() if p.name!='status.json'},
        'all_rows_untruncated':True,'all_assistant_turns_supervised':True,'real_tokenizer_audited':True,
        'human_review':'pending','ready_for_training':False,'training_started':False,'model_weights_loaded':False,'cuda_initialized':False}


def compare(directory):
    c.audit.cpu_gate(os.environ)
    manifest,episodes=verify()
    remote=c.REPORT/'remote'
    report=s.read(remote/'report.json')
    status=s.read(remote/'status.json')
    s.require(status['status']=='complete' and status['report_sha256']==s.sha(remote/'report.json')
        and report['passed'] and report['rows']==1200 and report['pairs']==600,'Incomplete remote audit')
    s.require(report['candidate_manifest_sha256']==s.sha(c.OUT/'manifest.json') and report['sources']==c.source_hashes()
        and report['versions']==s.STACK,'Remote binding/version changed')
    for key in ('all_rows_untruncated','all_assistant_turns_supervised','real_tokenizer_audited'): s.require(report[key] is True,'Audit check missing')
    for key in ('ready_for_training','training_started','model_weights_loaded','cuda_initialized'): s.require(report[key] is False,'Remote scope changed')
    for name,sha in report['files_sha256'].items(): s.require(Path(name).name==name and s.sha(remote/name)==sha,'Remote artifact changed')
    records=[json.loads(line) for line in (remote/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    s.require(len(records)==1200 and s.digest(records)==report['ordered_records_sha256'],'Incomplete records')
    for e,r in zip(episodes,records):
        s.require(r['id']==e['id'] and r['row_sha256']==s.digest(e['row']) and r['passed']
            and r['total_tokens']<=4096 and r['assistant_turns']==len(r['target_spans'])
            and r['permission_span']['text']==permission_text(e['spec']) and r['permission_span']['source_visible']
            and r['permission_labels_masked_as_expected'] and r['all_assistant_turns_supervised'] and r['public_source_spans_masked'], 'Unbound/incomplete record')
    sample=select_sample(episodes,{r['id']:r['total_tokens'] for r in records})
    s.require(sample==s.read(remote/'review-episodes.json'),'Stratified sample changed')
    return {'passed':True,'rows':1200,'sample_pairs':48,'remote_report_sha256':s.sha(remote/'report.json'),
        'candidate_manifest_sha256':s.sha(c.OUT/'manifest.json'),'real_tokenizer_audited':True,
        'comparison_kind':'Binding/consistency check of remote evidence; not independent tokenizer re-encoding',
        'human_review':'pending','ready_for_training':False,'training_started':False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('remote','compare'),required=True)
    parser.add_argument('--model-path',type=Path,default=Path(s.MODEL))
    args=parser.parse_args()
    directory=c.REPORT/args.stage
    s.fresh(directory)
    directory.mkdir(parents=True)
    s.write(directory/'status.json',{'status':'started','training_started':False})
    try:
        result=audit(directory,args.model_path) if args.stage=='remote' else compare(directory)
        s.write(directory/'report.json',result)
        s.update(directory/'status.json',{'status':'complete','report_sha256':s.sha(directory/'report.json'),'training_started':False})
        print(json.dumps({k:result[k] for k in ('passed','rows','human_review','ready_for_training','training_started')}),flush=True)
    except BaseException as exc:
        s.update(directory/'status.json',{'status':'failed','error':repr(exc),'training_started':False})
        raise


if __name__=='__main__': main()
