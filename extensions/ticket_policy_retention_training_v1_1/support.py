"""Independent retention recipe; CPU verification never imports CUDA/model code."""
from collections import Counter, defaultdict
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from extensions.ticket_training_v1_2 import support as old
from extensions.ticket_policy_diagnostic_v1_1 import core as diagnostic
from extensions.ticket_policy_retention_release_v1_1 import core as release

ROOT = old.ROOT
HERE = Path(__file__).resolve().parent
REPORT = ROOT / 'reports/ticket-policy-retention-training-v1.1'
RUN_NAME = 'runs/ticket-policy-retention-sft-v1.1'
RUN = ROOT / RUN_NAME
RELEASE = release.DATA
RELEASE_SHA = '171532e04b1ac8cc229ea11c476590b2a44dccdca0bd23d4c8d40384ea497885'
EVAL_SHA = 'fca28aeab1eaf27f5d10bed8bcb0647d5d9e479bef09e26e679601d21fd5684b'
START = '/konglingjie/sft_lora/lora_sft_toolcall_day1/runs/ticket-sft-v3-hermes-r8-v1.1/train/adapter'
START_SHA = '789144f12d35807f193974a8b753ada3e18327da64557e5551f96815553cb802'
START_CONFIG_SHA = '55b566d72adb5d90795db64ed76c33e5bff318ec032a735b593d4c25e6455bc2'
MODEL, STACK, POLICY = old.MODEL, old.STACK, diagnostic.POLICY
VARIANTS = ('ticket-sft-v3', 'ticket-policy-retention-v1.1')
COUNTS = {'smoke': 12, 'dev': 60, 'diagnostic': 72, 'targeted': 192}
CORPUS_SHA = 'bfd408fb8afec8db5fea986cf804aa82b21b26bfc8c514f160db26d62cc7d5f0'
BASE_EVIDENCE = ROOT / 'reports/ticket-policy-audit-v1/local/inputs/run-ticket-sft-v3.json'
BASE_EVIDENCE_SHA = '9617c1883f46a1b345d5059b8bbef6908d602d7aae81f5635589b8bcaf759b7e'
for _name in ('require','sha','digest','read','write','update','now','fresh','finite',
              'check_adapter','check_dataset','compare_tensor_sets','legacy'):
    globals()[_name] = getattr(old, _name)

def dev_gate():
    tasks,refs,meta=old.dev_gate()
    require(meta['corpus_manifest_sha256']==CORPUS_SHA and meta['frozen_tasks_sha256']==old.DEV_SHA,'Frozen DEV corpus/descriptor changed')
    return tasks,refs,meta

def private_module(name, relative):
    """New module instance, not mutation of old module globals/source."""
    spec = importlib.util.spec_from_file_location(__package__ + '.' + name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def config_for(stage):
    require(stage in ('preflight','train'), 'Unknown training stage')
    config = old.predecessor.config_for(stage)
    config.update(adapter_name_or_path=START, create_new_adapter=False,
        lora_target=','.join(sorted(old.TARGETS)), learning_rate=1e-5, num_train_epochs=1.0,
        output_dir=f'{RUN_NAME}/{stage}/adapter', dataset_dir=RELEASE.relative_to(ROOT).as_posix(),
        dataset=release.DATASET, save_total_limit=1, lr_scheduler_type='cosine', warmup_ratio=0.05)
    if stage == 'preflight':
        config.update(dataset_dir=f'{RUN_NAME}/preflight/data', dataset='ticket_policy_retention_preflight_v1_1')
    return config

def config_path(stage): return ROOT / f'configs/ticket_policy_retention_{stage}_v1_1.yaml'
def check_config(config, stage):
    require(config == config_for(stage), 'Changed recipe: no mixing/resume/new adapter/eval/extra flags')

def select_preflight(episodes, records):
    require(len(episodes) == len(records) == 1800, 'Incomplete training population')
    require([e['id'] for e in episodes] == [r['id'] for r in records], 'Token record ordering changed')
    pairs = defaultdict(list)
    lengths = {r['id']: r['total_tokens'] for r in records}
    for e in episodes: pairs[e['spec']['pair_id']].append(e)
    require(len(pairs) == 900 and all(len(es)==2 and {e['locale'] for e in es}=={'zh','en'} for es in pairs.values()), 'Broken bilingual pair')
    score = lambda pid: max(lengths[e['id']] for e in pairs[pid])
    rng, chosen, longest = random.Random(42), set(), {}
    for theme in sorted(release.QUOTAS):
        ids = sorted(pid for pid, es in pairs.items() if es[0]['spec']['theme']==theme)
        longest[theme] = sorted(ids, key=lambda pid:(-score(pid),pid))[0]
        if theme == 'authority':
            groups = defaultdict(list)
            for pid in ids: groups[pairs[pid][0]['spec']['contrast_group']].append(pid)
            require(len(groups)==90 and all(len(ps)==2 and {pairs[p][0]['spec']['allow_write'] for p in ps}=={False,True} for ps in groups.values()), 'Broken permission contrast group')
            selected = {pairs[longest[theme]][0]['spec']['contrast_group']}
            for profile in release.c.PROFILES:
                if any(pairs[groups[g][0]][0]['spec']['permission_profile']==profile for g in selected): continue
                selected.add(rng.choice(sorted(g for g in groups if pairs[groups[g][0]][0]['spec']['permission_profile']==profile)))
            selected.update(rng.sample(sorted(set(groups)-selected),4-len(selected)))
            chosen.update(pid for g in selected for pid in groups[g])
        else:
            selected = {longest[theme]}
            for profile in release.c.PROFILES:
                if any(pairs[p][0]['spec']['permission_profile']==profile for p in selected): continue
                selected.add(rng.choice(sorted(p for p in ids if pairs[p][0]['spec']['permission_profile']==profile)))
            selected.update(rng.sample(sorted(set(ids)-selected),8-len(selected)))
            chosen.update(selected)
    indices = [i for i,e in enumerate(episodes) if e['spec']['pair_id'] in chosen]
    require(len(indices)==96 and len(chosen)==48 and Counter(episodes[i]['spec']['theme'] for i in indices)=={k:16 for k in release.QUOTAS}, 'Preflight quotas differ')
    return indices, longest

def inputs(check_selection=True):
    episodes = read(release.c.OUT / 'episodes.json')
    records = [diagnostic.strict_json(line) for line in (RELEASE/'token-records.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = read(RELEASE / 'train.json')
    require(len(rows)==len(records)==len(episodes)==1800 and rows==[e['row'] for e in episodes], 'Released projection/count/order changed')
    for e,r in zip(episodes,records):
        require(e['id']==r['id'] and digest(e['row'])==r['row_sha256'] and r['passed'] is True, 'Episode/token evidence differs')
    require(sum(r['assistant_turns'] for r in records)==5202, 'Assistant denominator changed')
    indices,longest = select_preflight(episodes,records)
    selection = {'seed':42,'rows':96,'pairs':48,'episode_ids':[episodes[i]['id'] for i in indices],
        'longest_pair_by_category':longest,'subset_rows_digest':digest([rows[i] for i in indices]),
        'token_records_digest':digest([records[i] for i in indices]),'source_manifest_sha256':RELEASE_SHA}
    if check_selection: require(read(REPORT/'preflight-selection.json')==selection,'Frozen training-only selection changed')
    return episodes,records,indices,selection

def targeted_gate():
    require(sha(release.c.EVAL/'manifest.json')==EVAL_SHA,'Targeted manifest changed')
    manifest, gold = release.p.eval_gate(replay=False)
    tasks = read(release.c.EVAL/'tasks.json')
    require(len(tasks)==len(gold)==192 and tasks==[release.p.oracle(e) for e in gold],'Frozen targeted tasks changed')
    return tasks, gold, manifest

def sources():
    files = sorted(HERE.glob('*.py')) + [HERE/'README.md']
    files += [config_path(stage) for stage in ('preflight','train')]
    files += [ROOT/'scripts'/name for name in ('run_ticket_policy_retention_training_v1_1.py','package_ticket_policy_retention_training_v1_1.py','package_ticket_policy_retention_training_evidence_v1_1.py')]
    files += [ROOT/'tests/test_ticket_policy_retention_training_v1_1.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files}

def check_bindings(bindings):
    for name,expected in bindings.items():
        relative=Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe dependency path')
        require((ROOT/relative).is_file(), 'Missing prerequisite: '+name)
        require(sha(ROOT/relative)==expected,'Source/dependency changed: '+name)

def verify_release_cpu():
    # Old CPU gate remains unchanged; never temporarily change a GPU process's
    # environment or relax its import/GPU protection. Trace read prerequisites.
    environment,_=diagnostic.child_environment(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES='',HF_DATASETS_OFFLINE='1',PYTHONIOENCODING='utf-8')
    code=r'''
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(); reads=set()
def capture(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes)): return
    mode=args[1]
    if mode and any(c in mode for c in ('w','a','x','+')): return
    path=Path(args[0]).resolve()
    if path.is_relative_to(root) and path.is_file() and path.suffix not in ('.pyc','.zip'):
        relative=path.relative_to(root).as_posix()
        if 'final_test' in relative or 'final-test' in relative: raise ValueError('Forbidden read path')
        reads.add(path)
sys.addaudithook(capture)
from extensions.ticket_policy_audit_v1 import core as audit
audit.cpu_gate(__import__('os').environ)
sys.meta_path.insert(0,audit.CPUOnlyImports())
from extensions.ticket_policy_retention_release_v1_1 import core as release
result=release.verify()
for module in list(sys.modules.values()):
    value=getattr(module,'__file__',None)
    if value:
        path=Path(value).resolve()
        if path.is_relative_to(root) and path.is_file() and path.suffix=='.py': reads.add(path)
bindings={path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(reads)}
print('RETENTION_CPU_RESULT='+json.dumps({'verification':result,'dependencies':bindings},ensure_ascii=True))
'''
    result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT)],cwd=ROOT,env=environment,
        capture_output=True,text=True,encoding='utf-8',timeout=600)
    require(result.returncode==0,'CPU release verification failed; no GPU/model code executed:\n'+result.stdout+result.stderr)
    lines=[line for line in result.stdout.splitlines() if line.startswith('RETENTION_CPU_RESULT=')]
    require(len(lines)==1,'Missing/duplicate CPU verification receipt')
    proof=diagnostic.strict_json(lines[0].split('=',1)[1])
    require(proof['verification']['release_manifest_sha256']==RELEASE_SHA,'CPU receipt source differs')
    return proof

def local_gate():
    require(sha(RELEASE/'manifest.json')==RELEASE_SHA,'Wrong published release')
    verified=verify_release_cpu()['verification']
    for stage in ('preflight','train'): check_config(read(config_path(stage)),stage)
    inputs(); tasks,_,_=targeted_gate()
    descriptor=old.frozen_descriptor()
    probes=diagnostic.build_tasks(); diagnostic.validate_tasks(probes,descriptor['tasks'])
    require(digest(probes)=='05e1d7f3647b4d900da3686ef7e83ce62f79482de228d463c324c3c466909607','Diagnostic population changed')
    require(sha(BASE_EVIDENCE)==BASE_EVIDENCE_SHA,'Initial model provenance changed')
    return {'passed':True,'rows':1800,'pairs':900,'assistant_turns':5202,'expected_steps':225,
        'preflight_rows':96,'preflight_steps':10,'planned_evaluation_episodes':672,
        'release_manifest_sha256':RELEASE_SHA,'targeted_tasks_sha256':digest(tasks),
        'training_authorized':False,'training_started':False,'gpu_inference_started':False,
        'remote_runtime_and_weights':'pending_server_execution','old_final_test_read':False,
        'duplicate_metrics':verified['duplicate_metrics']}

def delivery_gate():
    d=read(REPORT/'delivery.json')
    require(d['sources']==sources(),'Training implementation/configuration changed')
    require(not d['training_authorized'] and not d['training_started'] and not d['gpu_inference_started'],'Delivery overstates execution')
    check_bindings(d['dependencies']); check_bindings(d['report_files_sha256'])
    return local_gate()

def dependencies(): return read(REPORT/'delivery.json')['dependencies']

def artifact_gate():
    require(sha(BASE_EVIDENCE)==BASE_EVIDENCE_SHA,'Base provenance changed')
    expected=read(BASE_EVIDENCE)['artifact_evidence']
    for name,value in expected['base_weights_sha256'].items():
        require(Path(name).name==name and sha(Path(MODEL)/name)==value,'Base weight/index differs: '+name)
    require(sha(Path(START)/'adapter_model.safetensors')==START_SHA and sha(Path(START)/'adapter_config.json')==START_CONFIG_SHA,'Wrong ticket-sft-v3 initializer')
    check_adapter(read(Path(START)/'adapter_config.json'))
    audit=read(RELEASE/'token-audit.json')
    for name,value in audit['tokenizer_asset_sha256'].items():
        require(Path(name).name==name and sha(Path(MODEL)/name)==value,'Tokenizer/config asset differs')
    import inspect
    from llamafactory.hparams import get_train_args
    from llamafactory.train.sft.workflow import run_sft
    from llamafactory.model.adapter import init_adapter
    from llamafactory.data.converter import SharegptDatasetConverter
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from transformers import Trainer
    pinned=ROOT/'LlamaFactory-ca75f1edf3cb50343ed1c98605141c3e22075b5f/src'
    require(Path(inspect.getsourcefile(get_train_args)).resolve().is_relative_to(pinned.resolve()),'Framework not pinned checkout')
    for obj in (get_train_args,run_sft,init_adapter,Trainer):
        require(sha(inspect.getsourcefile(obj))==expected['framework_source_sha256'][obj.__name__],'Framework source changed')
    for obj in (SharegptDatasetConverter,SupervisedDatasetProcessor):
        require(sha(inspect.getsourcefile(obj))==audit['framework_source_sha256'][obj.__name__],'Preprocessor source changed')
    return {'initializer_sha256':START_SHA,'initializer_config_sha256':START_CONFIG_SHA,
        'base_weights_sha256':expected['base_weights_sha256'],'framework_source_sha256':expected['framework_source_sha256'],
        'tokenizer_asset_sha256':audit['tokenizer_asset_sha256']}

def runtime_gate():
    diagnostic.environment_gate(os.environ)
    return old.runtime_gate()

def storage_gate():
    required=max(2*1024**3,16*(Path(START)/'adapter_model.safetensors').stat().st_size+256*1024**2)
    free=shutil.disk_usage(ROOT).free
    require(free>=required,'Insufficient disk; no recipe/checkpoint reduction')
    return {'free_bytes':free,'required_free_bytes':required}

def authorize_record(message):
    require(message=='授权启动训练及配对评测','Separate explicit execution authorization required')
    return {'user_message':message,'scope':'10-step preflight; 1800-row 225-step one-epoch continuation; 672 paired evaluation episodes',
        'provenance':'User CLI attestation; not independent authentication/signature',
        'delivery_sha256':sha(REPORT/'delivery.json'),'release_manifest_sha256':RELEASE_SHA,'initializer_sha256':START_SHA,
        'training_authorized':True,'gpu_inference_authorized':True,'created_at_utc':now()}

def execution_gate():
    record=read(RUN/'execution-authorization.json'); expected=authorize_record(record['user_message'])
    expected['created_at_utc']=record['created_at_utc']
    require(record==expected,'Execution authorization/source differs')
    return sha(RUN/'execution-authorization.json')

def active_launch_gate():
    require((RUN/'launch.lock').is_file() and read(RUN/'status.json')['status']=='started','Use authorized fresh-stage launcher')

def eval_directory(stage,variant=None):
    require(stage in COUNTS and (variant is None or variant in VARIANTS),'Unknown evaluation stage/adapter')
    return RUN/'eval'/stage/variant if variant else RUN/'eval'/stage
