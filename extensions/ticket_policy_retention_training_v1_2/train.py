"""Real pinned Trainer; never resume optimizer or reuse preflight weights."""
import gc
from pathlib import Path
from . import support as s

helpers=s.private_module('_train_helpers','extensions/ticket_policy_repair_training_v1/train.py')
initial_gate=helpers.initial_gate
reload_arguments=helpers.reload_arguments
check_updated_tensors=helpers.check_updated_tensors

def state_gate(state,metrics,stage):
    steps=10 if stage=='preflight' else 225
    s.require(state.get('global_step')==state.get('max_steps')==steps,'Wrong completed step budget')
    epoch=state.get('epoch')
    s.require(s.finite(epoch),'Missing/nonfinite epoch')
    s.require(abs(epoch-1.0)<1e-6 if stage=='train' else 0<epoch<1,'Wrong epoch; never force preflight to one epoch')
    s.require(s.finite(metrics.get('train_loss')),'Missing/nonfinite loss')
    for item in state.get('log_history',[]):
        for key in ('loss','train_loss','grad_norm','learning_rate'):
            if key in item: s.require(s.finite(item[key]),'Nonfinite '+key)

def prepare_inputs(stage,directory):
    episodes,records,indices,selection=s.inputs()
    if stage=='train': return records
    data=directory/'data'; s.fresh(data); data.mkdir()
    s.write(data/'train.json',[episodes[i]['row'] for i in indices])
    registration=s.read(s.RELEASE/'dataset_info.json')[s.release.DATASET]
    s.write(data/'dataset_info.json',{'ticket_policy_retention_preflight_v1_2':registration})
    s.write(directory/'sample.json',selection)
    return [records[i] for i in indices]

def checkpoint_gate(adapter,stage):
    step=10 if stage=='preflight' else 225
    paths={p.name:p for p in adapter.glob('checkpoint-*') if p.is_dir()}
    s.require(set(paths)=={f'checkpoint-{step}'},'Missing/unexpected checkpoint')
    directory=paths[f'checkpoint-{step}']
    state_gate(s.read(directory/'trainer_state.json'),s.read(adapter/'all_results.json'),stage)
    for name in ('adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','trainer_state.json'):
        s.require((directory/name).is_file() and (directory/name).stat().st_size>0,'Incomplete checkpoint: '+name)
    s.check_adapter(s.read(directory/'adapter_config.json'))
    s.require(s.sha(directory/'adapter_model.safetensors')==s.sha(adapter/'adapter_model.safetensors'),'Checkpoint/final adapter mismatch')
    return {directory.name:{p.name:s.sha(p) for p in sorted(directory.iterdir()) if p.is_file()}}

def verify_result(directory,stage):
    adapter=directory/'adapter'; s.check_config(s.read(directory/'config.json'),stage)
    state=s.read(adapter/'trainer_state.json'); metrics=s.read(adapter/'all_results.json')
    state_gate(state,metrics,stage); s.check_adapter(s.read(adapter/'adapter_config.json'))
    episodes,all_records,indices,_=s.inputs()
    records=all_records if stage=='train' else [all_records[i] for i in indices]
    audit=s.read(directory/'training-input-audit.json')
    expected={'passed':True,'rows':len(records),'assistant_turns':sum(r['assistant_turns'] for r in records),'ordered_audit_sha256':s.digest(records)}
    s.require(audit==expected,'Actual Trainer tokens/labels/source mismatch')
    initial=s.read(directory/'initialization.json')
    s.require(initial['passed'] and initial['optimizer_fresh'] and initial['scheduler_fresh'] and initial['global_step']==0 and initial['initializer_sha256']==s.START_SHA,'Wrong initial adapter/optimizer')
    s.require(s.sha(Path(s.START)/'adapter_model.safetensors')==s.START_SHA and s.sha(Path(s.START)/'adapter_config.json')==s.START_CONFIG_SHA,'Original initializer changed')
    weights=adapter/'adapter_model.safetensors'
    s.require(weights.is_file() and weights.stat().st_size>0 and s.sha(weights)!=s.START_SHA,'Weights missing/unchanged')
    from safetensors.torch import load_file
    changed=check_updated_tensors(load_file(str(Path(s.START)/'adapter_model.safetensors'),device='cpu'),load_file(str(weights),device='cpu'))
    return {'passed':True,**expected,'global_step':state['global_step'],'epoch':state['epoch'],'train_loss':metrics['train_loss'],
        'adapter_sha256':s.sha(weights),'adapter_config_sha256':s.sha(adapter/'adapter_config.json'),
        'adapter_bytes':weights.stat().st_size,'initializer_unchanged':True,**changed,'checkpoint_files_sha256':checkpoint_gate(adapter,stage)}

def check_completed(stage):
    directory=s.RUN/stage; run=s.read(directory/'run.json')
    s.require(run['status']=='complete' and run['stage']==stage and run['config']==s.config_for(stage),'Training stage incomplete/config changed')
    s.require(run['allocated_gpu']['selection']==s.GPU and run['allocated_gpu']['uuid'].startswith('GPU-'), 'Training ran on wrong physical GPU')
    s.require(run['sources']==s.sources() and run['dependencies']==s.dependencies() and run['authorization_sha256']==s.execution_gate() and run['release_manifest_sha256']==s.RELEASE_SHA,'Training source/authorization mismatch')
    names={'config.json','initialization.json','training-input-audit.json','reload-input.json','reload.json','verification.json',
        'adapter/adapter_config.json','adapter/adapter_model.safetensors','adapter/trainer_state.json','adapter/all_results.json'}
    if stage=='preflight': names|={'sample.json','data/train.json','data/dataset_info.json'}
    s.require(set(run['outputs_sha256'])==names,'Incomplete training evidence')
    for name,h in run['outputs_sha256'].items(): s.require(s.sha(directory/name)==h,'Training output changed: '+name)
    result=verify_result(directory,stage); reload=s.read(directory/'reload.json')
    s.require(reload['passed'] and reload['finite_forward'] and reload['adapter_sha256']==result['adapter_sha256'],'Reload missing/wrong adapter')
    s.require(s.read(directory/'verification.json')=={**result,'reload':reload},'Training acceptance changed')
    return run

def execute(stage):
    s.delivery_gate(); s.active_launch_gate(); authorization=s.execution_gate()
    config=s.read(s.config_path(stage)); s.check_config(config,stage)
    directory=s.RUN/stage; s.fresh(directory)
    if stage=='train': check_completed('preflight')
    storage=s.storage_gate(); versions,gpu=s.runtime_gate(); artifacts=s.artifact_gate(); dev_meta=s.dev_gate()[2]
    directory.mkdir()
    run={'status':'started','stage':stage,'started_at_utc':s.now(),'versions':versions,'allocated_gpu':gpu,
        'storage_admission':storage,'config':config,'sources':s.sources(),'dependencies':s.dependencies(),
        'authorization_sha256':authorization,'release_manifest_sha256':s.RELEASE_SHA,'artifacts':artifacts,'frozen_dev':dev_meta,
        'selection':'One epoch/final adapter; no DEV loss or checkpoint selection; fresh original initializer'}
    s.write(directory/'run.json',run)
    try:
        s.write(directory/'config.json',config); records=prepare_inputs(stage,directory)
        from transformers import set_seed
        from llamafactory.hparams import get_train_args
        from llamafactory.train.sft.workflow import run_sft
        set_seed(42); args=get_train_args(config)
        s.require(args[2].resume_from_checkpoint is None,'Optimizer/checkpoint resume forbidden')
        run_sft(*args,callbacks=[helpers.build_callback(directory,records,10 if stage=='preflight' else 225)])
        import torch
        gc.collect(); torch.cuda.empty_cache()
        result=verify_result(directory,stage)
        reload=helpers.reload_check(directory/'adapter',s.read(directory/'reload-input.json')['input_ids'])
        s.write(directory/'reload.json',reload); s.write(directory/'verification.json',{**result,'reload':reload})
        s.require(s.sources()==run['sources'] and s.artifact_gate()==artifacts and s.dev_gate()[2]==dev_meta,'Sources changed during training')
        s.delivery_gate()
        names=['config.json','initialization.json','training-input-audit.json','reload-input.json','reload.json','verification.json',
            'adapter/adapter_config.json','adapter/adapter_model.safetensors','adapter/trainer_state.json','adapter/all_results.json']
        if stage=='preflight': names+=['sample.json','data/train.json','data/dataset_info.json']
        run.update(status='complete',finished_at_utc=s.now(),outputs_sha256={n:s.sha(directory/n) for n in names})
        s.update(directory/'run.json',run); print(result,flush=True)
    except BaseException as exc:
        run.update(status='failed',finished_at_utc=s.now(),error=f'{type(exc).__name__}: {exc}')
        s.update(directory/'run.json',run); raise

if __name__=='__main__':
    import argparse,os,signal
    def interrupted(signum,frame): raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage',choices=('preflight','train'),required=True)
    os.chdir(s.ROOT); execute(parser.parse_args().stage)
