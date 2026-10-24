"""Read whitelisted tar bytes; verify uploaded evidence without model execution.

HTML is kept as data and never executed. A private copy of the hash-bound CPU
comparison function points only its evidence directory at the uploaded copy.
The original module globals, sources and historical outputs remain unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import re
import sys
import tarfile
import traceback
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extensions.ticket_policy_retention_build_v1 import core as c,pipeline as p,remote as r
from extensions.ticket_policy_audit_v1 import core as audit

PREFIX='reports/ticket-policy-retention-build-v1/'
EXPECTED={'remote/'+n for n in ('report.json','status.json','started.json','records.jsonl','review-episodes.json','review-pairs.json','review.md','review.html')}|{
  'compare/'+n for n in ('report.json','status.json','started.json')}|{'logs/'+n for n in ('remote-environment.json','compare-environment.json','remote.log','compare.log')}
OUT=ROOT/'reports/ticket-policy-retention-upload-review-v1'

def bytes_sha(data): return hashlib.sha256(data).hexdigest()
def parse(data): return audit.diagnostic.strict_json(data.decode('utf-8'))

def read_archive(path):
    c.require(path.stat().st_size<20_000_000,'Compressed evidence too large')
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers()
        c.require(len(members)<=32 and len({x.name for x in members})==len(members) and sum(x.size for x in members)<=50_000_000,'Duplicate/oversized archive')
        files={}
        for m in members:
            path=PurePosixPath(m.name)
            c.require(not path.is_absolute() and '..' not in path.parts and '\\' not in m.name and ':' not in m.name,'Unsafe archive path')
            c.require(m.name.startswith(PREFIX) and (m.isfile() or m.isdir()),'Unexpected root/link/member type')
            rel=m.name[len(PREFIX):].rstrip('/')
            if m.isdir(): c.require(rel in ('remote','compare','logs'),'Unexpected directory')
            else:
                c.require(rel in EXPECTED,'Unexpected file: '+rel)
                with archive.extractfile(m) as stream: files[rel]=stream.read()
                c.require(len(files[rel])==m.size,'Truncated member')
        c.require(set(files)==EXPECTED,'Missing evidence member')
        return files

def stage_logs(files):
    for stage,review in (('remote','pending'),('compare','pending_AI_assisted_then_user_acceptance')):
        report=parse(files[stage+'/report.json']); status=parse(files[stage+'/status.json'])
        c.require(status['status']=='complete' and status['report_sha256']==bytes_sha(files[stage+'/report.json']),'Incomplete stage: '+stage)
        started=parse(files[stage+'/started.json'])
        c.require(started['stage']==stage and started['training_started'] is False,'Wrong stage start')
        c.require(report['passed'] is True and report['sample_review']==review and report['rows']==1800 and report['ready_for_training'] is False and report['training_started'] is False,'Wrong uploaded stage scope')
        env=parse(files['logs/'+stage+'-environment.json'])
        c.require(env['CUDA_VISIBLE_DEVICES']=='' and env['offline'] is True and env['training_started'] is False and env['model_weights_loaded'] is False and env['python'].startswith('3.11.') and isinstance(env['removed_distributed_keys'],list),'Unsafe/unpinned reported runtime')
        log=files['logs/'+stage+'.log'].decode('utf-8')
        c.require(parse(log.splitlines()[-1].encode())=={'passed':True,'rows':1800,'sample_pairs':60,'sample_review':review,'ready_for_training':False,'training_started':False},'Incomplete stage log')
        if stage=='remote':
            progress=list(map(int,re.findall(r'Actual CPU tokenizer/labels: (\d+)/1800',log)))
            c.require(progress==list(range(100,1801,100)),'Missing/duplicate real CPU audit progress')
    c.require(parse(files['compare/report.json'])['remote_report_sha256']==bytes_sha(files['remote/report.json']),'Comparison source mismatch')

def private_compare(evidence):
    source=c.HERE/'remote.py'
    c.require(c.sha(source)==c.read(c.OUT/'manifest.json')['sources_sha256']['extensions/ticket_policy_retention_build_v1/remote.py'],'Changed bound comparison source')
    name='extensions.ticket_policy_retention_build_v1._uploaded_compare'
    loader=importlib.util.spec_from_file_location(name,source)
    module=importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
    module.c=SimpleNamespace(**{k:v for k,v in vars(c).items() if not k.startswith('__')})
    module.c.REPORT=evidence
    # Its p.verify() still points at the original confirmed local candidate.
    return module.compare(evidence/'unused-read-only-target')

def view(sample):
    lines=['# AI 辅助语义审核阅读视图：60 对','', '原始正文、双语请求、回答与真实调用；不是人工批准。','']
    for zh,en in zip(sample[::2],sample[1::2]):
        s=zh['spec']
        lines+=['## '+s['pair_id']+' · '+s['theme']+'/'+s['variant']+' · '+s['permission_profile']+' · writes='+str(s['allow_write']),'']
        for e in (zh,en):
            lines+=['### '+e['locale'],'','权限：'+e['row']['system'].split('\n')[-1],'']
            lines +=[m['from']+': '+m['value'] for m in e['row']['conversations'] if m['from'] in ('human','gpt')]
        lines+=['','真实调用（双语相同）：']
        for event in zh['events']:
            lines+=[c.canonical(event['call'])+' → '+c.canonical(event['response'])]
        if zh['external_events']: lines+=['外部事件：'+c.canonical(zh['external_events'])]
        lines+=['']
    return '\n'.join(lines)+'\n'

def verify(archive,out):
    p.fresh(out); files=read_archive(archive); stage_logs(files)
    out.mkdir(parents=True)
    p.write(out/'started.json',{'stage':'uploaded_evidence_review','archive_sha256':c.sha(archive),'started_at':p.now(),'training_started':False})
    try:
        for name,data in files.items():
            target=out/'evidence'/name
            c.require(target.resolve().is_relative_to((out/'evidence').resolve()),'Extraction escaped target')
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream: stream.write(data)
        env,_=audit.cpu_environment(os.environ); os.environ.clear(); os.environ.update(env)
        reproduced=private_compare(out/'evidence')
        c.require(reproduced==parse(files['compare/report.json']),'Uploaded comparison differs from local recomputation')
        sample=parse(files['remote/review-episodes.json'])
        p.text(out/'semantic-review-view.md',view(sample))
        records=[parse(line.encode()) for line in files['remote/records.jsonl'].decode('utf-8').splitlines()]
        remote=parse(files['remote/report.json'])
        result={**reproduced,'review_type':'AI_assisted_evidence_consistency_review',
          'archive_sha256':c.sha(archive),'member_sha256':{n:bytes_sha(data) for n,data in files.items()},
          'sample_pair_ids':[e['spec']['pair_id'] for e in sample[::2]],
          'all_1800_records_verified':True,'all_1800_public_trajectories_replayed':True,
          'frozen_192_targeted_oracles_replayed':True,'fixed_60_pair_sample_recomputed':True,
          'token_audit':'Uploaded remote real tokenizer evidence verified; no independent local tokenizer rerun',
          'versions':remote['versions'],'remote_python':remote['python'],
          'total_supervised_tokens':sum(x['supervised_tokens'] for x in records),
          'new_review_source_sha256':c.sha(Path(__file__)),'read_view_sha256':c.sha(out/'semantic-review-view.md'),
          'AI_semantic_review':'pending_actual_reading','human_acceptance':'pending',
          'per_row_human_inspection_attested':False,'ready_for_training':False,'gpu_inference_started':False,'old_final_test_read':False}
        p.write(out/'evidence-review.json',result)
        p.write(out/'evidence-status.json',{'status':'complete','evidence_review_sha256':c.sha(out/'evidence-review.json'),'training_started':False})
        print(json.dumps({k:result[k] for k in ('passed','rows','max_tokens','assistant_turns','sample_pairs','sample_rows','human_acceptance','ready_for_training','training_started')},indent=2))
        return result
    except BaseException as exc:
        p.text(out/'failure.log',traceback.format_exc()); p.write(out/'failure.json',{'error':repr(exc),'training_started':False}); raise

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--archive',type=Path,required=True); parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args(); verify(args.archive,args.out)
if __name__=='__main__': main()
