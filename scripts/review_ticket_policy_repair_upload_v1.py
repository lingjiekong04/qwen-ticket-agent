"""Review uploaded CPU evidence; no tokenizer rerun, model, training or approval."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extensions.ticket_policy_repair_build_v1 import core as c
from extensions.ticket_policy_repair_build_v1.pipeline import verify,select_sample,render_review
from extensions.ticket_policy_repair_build_v1.remote import permission_text
s=c.s
EXPECTED={'remote/records.jsonl','remote/report.json','remote/status.json','remote/review-episodes.json',
    'remote/review-pairs.json','remote/review.html','remote/review.md','compare/report.json','compare/status.json',
    'logs/remote-environment.json','logs/compare-environment.json','logs/remote.log','logs/compare.log'}


def sha(data): return hashlib.sha256(data).hexdigest()
def parse(data): return c.audit.diagnostic.strict_json(data.decode('utf-8'))
def hash_value(value): return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def read_archive(path):
    with zipfile.ZipFile(path) as archive:
        infos=archive.infolist()
        s.require(len(infos)<=32 and len({i.filename for i in infos})==len(infos)
            and sum(i.file_size for i in infos)<20_000_000,'Duplicate/oversized archive')
        s.require(archive.testzip() is None,'Bad ZIP CRC')
        files={}
        for item in infos:
            name=PurePosixPath(item.filename)
            s.require(not name.is_absolute() and '..' not in name.parts and '\\' not in item.filename
                and ':' not in item.filename and (item.external_attr>>16)&0o170000!=0o120000,'Unsafe ZIP member')
            if not item.is_dir():
                s.require(item.filename.startswith('3/'),'Unexpected archive root')
                relative=item.filename[2:]
                s.require(relative in EXPECTED,'Unexpected file: '+relative)
                files[relative]=archive.read(item.filename)
        s.require(set(files)==EXPECTED,'Missing uploaded evidence')
        return files


def check_records(records,episodes):
    s.require(len(records)==len(episodes)==1200 and len({r['id'] for r in records})==1200,'Incomplete/duplicate record population')
    patterns=defaultdict(set)
    for e,r in zip(episodes,records):
        spec=e['spec']
        s.require(r['id']==e['id'] and r['pair_id']==spec['pair_id'] and r['row_sha256']==s.digest(e['row']),'Record input/identity mismatch')
        s.require(all(r[k]==(e['locale'] if k=='locale' else spec[k]) for k in ('category','variant','profile','locale')),'Record metadata mismatch')
        s.require(r['passed'] is True and type(r['total_tokens']) is int and 0<r['total_tokens']<=4096
            and hash_value(r['input_ids_sha256']) and hash_value(r['labels_sha256']),'Invalid encoding record')
        assistant=[m for m in e['row']['conversations'] if m['from'] in {'gpt','function_call'}]
        s.require(r['assistant_turns']==len(assistant)==len(r['target_spans']) and r['converted_turns']==len(e['row']['conversations']),'Dropped/extra turns')
        previous=0
        counts=Counter()
        for span,message in zip(r['target_spans'],assistant):
            start,end,count=span['start'],span['end'],span['target_tokens']
            s.require(all(type(v) is int for v in (start,end,count)) and previous<start<end<=r['total_tokens']
                and end-start==count,'Bad assistant target span')
            counts[message['from']]+=count
            previous=end
        s.require(previous==r['total_tokens'] and r['supervised_tokens']==sum(counts.values())
            and r['call_target_tokens']==counts['function_call'] and r['text_target_tokens']==counts['gpt'],'Target accounting mismatch')
        span=r['permission_span']
        s.require(span['text']==permission_text(spec) and c.system(spec,e['locale']).count(span['text'])==1
            and span['source_visible'] is True and type(span['start']) is int and type(span['end']) is int
            and 0<=span['start']<span['end']<=r['target_spans'][0]['start'],'Permission missing/ambiguous/outside first source')
        ids=span['token_ids']
        s.require(len(ids)==span['end']-span['start'] and all(type(x) is int and 0<=x<200000 for x in ids)
            and s.digest(ids)==span['token_ids_sha256'],'Invalid permission token evidence')
        s.require(all(r[k] is True for k in ('permission_labels_masked_as_expected','all_assistant_turns_supervised','public_source_spans_masked')),'Incomplete label checks')
        patterns[span['text']].add(tuple(ids))
    s.require(all(len(v)==1 for v in patterns.values()),'Same permission text has inconsistent token spans')


def check_evidence(files,episodes):
    remote,compare=(parse(files[f'{name}/report.json']) for name in ('remote','compare'))
    for stage,report in (('remote',remote),('compare',compare)):
        status=parse(files[stage+'/status.json'])
        s.require(status['status']=='complete' and status['report_sha256']==sha(files[stage+'/report.json'])
            and status['training_started'] is False and report['passed'] is True and report['rows']==1200,'Incomplete/unbound stage')
        s.require(report['human_review']=='pending' and report['ready_for_training'] is False and report['training_started'] is False
            and report['real_tokenizer_audited'] is True,'Stage scope changed')
        env=parse(files[f'logs/{stage}-environment.json'])
        s.require(env['CUDA_VISIBLE_DEVICES']=='' and env['offline'] is True and env['training_started'] is False
            and env['model_weights_loaded'] is False and isinstance(env['removed_distributed_keys'],list),'Reported environment changed')
        log=files[f'logs/{stage}.log'].decode('utf-8')
        s.require(parse(log.splitlines()[-1].encode())=={'passed':True,'rows':1200,'human_review':'pending','ready_for_training':False,'training_started':False},'Incomplete stage log')
    s.require(remote['candidate_manifest_sha256']==compare['candidate_manifest_sha256']==s.sha(c.OUT/'manifest.json')
        and remote['sources']==c.source_hashes() and remote['versions']==s.STACK
        and remote['python_requirement']=='3.11' and remote['model_path']==s.MODEL,'Source/runtime binding mismatch')
    prior=s.read(s.RELEASE/'token-audit.json')
    s.require(remote['tokenizer_asset_sha256']==prior['tokenizer_asset_sha256']
        and remote['framework_source_sha256']==prior['framework_source_sha256'],'Tokenizer/framework sources differ')
    s.require(all(remote[k] is True for k in ('all_rows_untruncated','all_assistant_turns_supervised'))
        and remote['model_weights_loaded'] is False and remote['cuda_initialized'] is False,'Remote checks/scope incomplete')
    s.require(set(remote['files_sha256'])=={'records.jsonl','review-pairs.json','review-episodes.json','review.html','review.md'},'Unexpected artifact bindings')
    for name,expected in remote['files_sha256'].items(): s.require(sha(files['remote/'+name])==expected,'Changed remote artifact: '+name)
    s.require(compare['remote_report_sha256']==sha(files['remote/report.json']) and compare['comparison_kind']=='Binding/consistency check of remote evidence; not independent tokenizer re-encoding','Comparison binding differs')
    records=[parse(line.encode()) for line in files['remote/records.jsonl'].decode().splitlines()]
    check_records(records,episodes)
    s.require(remote['records_sha256']==sha(files['remote/records.jsonl']) and remote['ordered_records_sha256']==s.digest(records),'Records digest mismatch')
    s.require(remote['max_tokens']==max(r['total_tokens'] for r in records) and remote['assistant_turns']==sum(r['assistant_turns'] for r in records),'Summary mismatch')
    coverage=defaultdict(lambda:defaultdict(int))
    for r in records:
        key=tuple(r[k] for k in ('category','locale','profile','variant'))
        for field in ('assistant_turns','total_tokens','supervised_tokens','call_target_tokens','text_target_tokens'): coverage[key][field]+=r[field]
        coverage[key]['rows']+=1
    wanted=[dict(zip(('category','locale','profile','variant'),k),**v) for k,v in sorted(coverage.items())]
    s.require(remote['coverage']==wanted,'Coverage differs from individual records')
    s.require(list(map(int,re.findall(r'CPU repair tokenizer/label audit: (\d+)/1200',files['logs/remote.log'].decode())))==list(range(100,1201,100)),'Progress missing')
    sample=select_sample(episodes,{r['id']:r['total_tokens'] for r in records})
    s.require(sample==parse(files['remote/review-episodes.json']) and [e['spec']['pair_id'] for e in sample if e['locale']=='zh']==parse(files['remote/review-pairs.json'])
        and remote['sample_pairs']==compare['sample_pairs']==48 and remote['sample_rows']==96,'Sample not reproducible')
    md,page=render_review(sample,'修复候选正式抽检：48 对 / 96 条（真实 token 长度分层）')
    s.require(md.encode()==files['remote/review.md'] and page.encode()==files['remote/review.html'],'Review display changed')
    return remote,records,sample


def review_view(sample):
    lines=['# 正式抽检：AI 阅读视图（48 对）','', '仅压缩重复参数展示；原完整审核页和 JSON 保留，非人工批准。','']
    for zh,en in zip(sample[::2],sample[1::2]):
        sp=zh['spec']
        lines+=['## '+sp['pair_id']+' · '+sp['category']+'/'+sp['variant']+' · '+sp['profile']+' · allow_write='+str(sp['policy']['allow_write']), '']
        if sp['category']!='priority': lines+=['业务字段：'+c.audit.canonical(sp['fields']),'']
        for e in (zh,en):
            lines+=['### '+e['locale'],'']
            for m in e['row']['conversations']:
                if m['from'] in {'human','gpt'}:
                    value=m['value']
                    if m['from']=='human' and sp['category']!='priority':
                        for key in ('title','description'): value=value.replace(c.audit.canonical(sp['fields'][key]),'{'+key+'}')
                    lines+=[m['from']+': '+value,'']
            calls=[]
            for event in e['events']:
                call,response=event['call'],event['response']
                brief=call['name']+' '+c.audit.canonical(call['arguments'])+' → '
                brief+=c.audit.canonical(response['result']) if response['ok'] else response['error']['code']
                calls.append(brief)
            lines+=['调用：'+(' | '.join(calls) if calls else '无调用'),'']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    s.fresh(args.out)
    args.out.mkdir(parents=True)
    s.write(args.out/'status.json',{'status':'started','human_review':'pending','training_started':False})
    try:
        _,episodes=verify()
        files=read_archive(args.archive)
        remote,records,sample=check_evidence(files,episodes)
        # Preserve only whitelisted bytes, never execute HTML/scripts from uploads.
        for name,data in files.items():
            path=args.out/'evidence'/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(data)
        (args.out/'semantic-review-view.md').write_text(review_view(sample),encoding='utf-8')
        result={'passed':True,'review_type':'AI-assisted evidence consistency review','human_review':'pending',
            'archive_sha256':s.sha(args.archive),'member_sha256':{k:sha(v) for k,v in files.items()},
            'candidate_manifest_sha256':s.sha(c.OUT/'manifest.json'),'rows':1200,'pairs':600,
            'max_tokens':remote['max_tokens'],'assistant_turns':remote['assistant_turns'],'sample_pairs':48,'sample_rows':96,
            'sample_pair_ids':[e['spec']['pair_id'] for e in sample if e['locale']=='zh'],
            'local_candidate_replay':'passed_all_1200','remote_record_binding':'matched_all_1200','formal_sample_recomputed':'matched',
            'scope':'Remote CPU audit evidence verified; not a local independent tokenizer rerun or signed execution attestation',
            'ready_for_training':False,'training_started':False,'model_weights_loaded_locally':False,
            'limits':['Full token sequences/labels are not included; span/accounting/hash evidence and bound remote assertions were checked.',
                'Actual Python runtime is enforced by hash-bound code; this report is not independent runtime authentication.',
                'No model performance, generalization, GPU inference or human approval is established.']}
        s.write(args.out/'evidence-review.json',result)
        s.update(args.out/'status.json',{'status':'complete','evidence_review_sha256':s.sha(args.out/'evidence-review.json'),'human_review':'pending','training_started':False})
        print(json.dumps({k:result[k] for k in ('passed','rows','max_tokens','assistant_turns','sample_pairs','human_review','training_started')}))
    except BaseException as exc:
        s.update(args.out/'status.json',{'status':'failed','error':repr(exc),'training_started':False})
        raise


if __name__=='__main__': main()
