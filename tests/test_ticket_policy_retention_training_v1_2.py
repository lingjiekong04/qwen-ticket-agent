"""CPU tests: no weights, real tokenizer, CUDA or old final test."""
from collections import Counter
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from extensions.ticket_policy_retention_training_v1_2 import support as s,train,targeted as t,evaluate as ev

class TrainingGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.episodes,cls.records,cls.indices,cls.selection=s.inputs(check_selection=False)
    def test_recipe(self):
        c=s.config_for('train'); self.assertEqual(c['learning_rate'],1e-5); self.assertEqual(c['num_train_epochs'],1)
        self.assertEqual(1800//c['gradient_accumulation_steps'],225); self.assertEqual(s.COUNTS,{'smoke':12,'dev':60,'diagnostic':72,'targeted':192})
    def test_config_mutations(self):
        for key,value in {'adapter_name_or_path':'wrong','create_new_adapter':True,'resume_from_checkpoint':'old',
            'dataset':'dev','train_on_prompt':True,'mask_history':True,'packing':True,'num_train_epochs':2,
            'learning_rate':2e-5,'cutoff_len':2048,'eval_dataset':'dev','load_best_model_at_end':True}.items():
            with self.subTest(key=key),self.assertRaises(ValueError): s.check_config({**s.config_for('train'),key:value},'train')
    def test_preflight_selection(self):
        selected=[self.episodes[i] for i in self.indices]
        self.assertEqual(len(selected),96); self.assertEqual(len({e['spec']['pair_id'] for e in selected}),48)
        self.assertEqual(Counter(e['spec']['theme'] for e in selected),{k:16 for k in s.release.QUOTAS})
        for theme in s.release.QUOTAS:
            self.assertEqual({e['spec']['permission_profile'] for e in selected if e['spec']['theme']==theme},set(s.release.c.PROFILES))
            self.assertIn(self.selection['longest_pair_by_category'][theme],{e['spec']['pair_id'] for e in selected})
        groups=Counter(e['spec']['contrast_group'] for e in selected if e['spec']['theme']=='authority')
        self.assertEqual(set(groups.values()),{4})
        self.assertEqual(s.select_preflight(self.episodes,self.records)[0],self.indices)
    def test_no_drop_reorder(self):
        with self.assertRaises(ValueError): s.select_preflight(self.episodes,self.records[:-1])
        with self.assertRaises(ValueError): s.select_preflight(self.episodes,list(reversed(self.records)))
    def test_bad_hash_missing_path(self):
        with self.assertRaises(ValueError): s.check_bindings({'extensions/ticket_policy_retention_training_v1_2/support.py':'bad'})
        with self.assertRaises(ValueError): s.check_bindings({'reports/absent-retention-dependency.json':'bad'})
        with self.assertRaises(ValueError): s.check_bindings({'../outside':'bad'})
    def test_bad_release_hash(self):
        original_sha=s.sha
        with patch.object(s,'sha',side_effect=lambda p:'wrong' if Path(p)==s.RELEASE/'manifest.json' else original_sha(p)),self.assertRaisesRegex(ValueError,'Wrong published release'): s.local_gate()
    def test_wrong_frozen_dev_manifest(self):
        with patch.object(s.old,'dev_gate',return_value=([],[],{'corpus_manifest_sha256':'wrong','frozen_tasks_sha256':s.old.DEV_SHA})),self.assertRaises(ValueError): s.dev_gate()
    def test_cpu_verify_no_model_import(self):
        import ast
        for node in ast.walk(ast.parse(Path(s.__file__).read_text(encoding='utf-8'))):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ('local_gate','delivery_gate','inputs'):
                imports=[x.module for x in ast.walk(node) if isinstance(x,ast.ImportFrom)]
                self.assertFalse(any(x and (x.startswith('torch') or x.startswith('llamafactory')) for x in imports))
    def test_cpu_release_isolated_from_gpu_parent(self):
        from types import SimpleNamespace
        import os,json
        proof={'verification':{'release_manifest_sha256':s.RELEASE_SHA},'dependencies':{}}
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1','RANK':'0','WORLD_SIZE':'1'}),patch.object(s.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='RETENTION_CPU_RESULT='+json.dumps(proof)+'\n',stderr='')) as process:
            self.assertEqual(s.verify_release_cpu(),proof)
            env=process.call_args.kwargs['env']; self.assertEqual(env['CUDA_VISIBLE_DEVICES'],''); self.assertEqual(env['HF_DATASETS_OFFLINE'],'1')
            self.assertNotIn('RANK',env); self.assertNotIn('WORLD_SIZE',env)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'1'); self.assertEqual(os.environ['RANK'],'0')
    def test_initial_optimizer_resume(self):
        from types import SimpleNamespace as N
        train.initial_gate(N(global_step=0,max_steps=225),N(resume_from_checkpoint=None),N(state={}),N(last_epoch=0),225)
        for state,args,opt,sched in [(N(global_step=1,max_steps=225),N(resume_from_checkpoint=None),N(state={}),N(last_epoch=0)),
            (N(global_step=0,max_steps=225),N(resume_from_checkpoint='old'),N(state={}),N(last_epoch=0)),
            (N(global_step=0,max_steps=225),N(resume_from_checkpoint=None),N(state={'old':1}),N(last_epoch=0))]:
            with self.assertRaises(ValueError): train.initial_gate(state,args,opt,sched,225)
    def test_state_loss_epochs(self):
        train.state_gate({'global_step':225,'max_steps':225,'epoch':1,'log_history':[{'loss':0.5}]},{'train_loss':0.5},'train')
        train.state_gate({'global_step':10,'max_steps':10,'epoch':80/96},{'train_loss':0.5},'preflight')
        for state,metrics,stage in [({'global_step':224,'max_steps':225,'epoch':1},{'train_loss':0.5},'train'),
            ({'global_step':225,'max_steps':225,'epoch':2},{'train_loss':0.5},'train'),
            ({'global_step':225,'max_steps':225,'epoch':1},{'train_loss':float('nan')},'train'),
            ({'global_step':225,'max_steps':225,'epoch':1,'log_history':[{'grad_norm':float('inf')}]},{'train_loss':0.5},'train'),
            ({'global_step':10,'max_steps':10,'epoch':1},{'train_loss':0.5},'preflight')]:
            with self.subTest(stage=stage),self.assertRaises(ValueError): train.state_gate(state,metrics,stage)
    def test_existing_outputs(self):
        with tempfile.TemporaryDirectory() as d,self.assertRaises(ValueError): s.fresh(Path(d))
    def test_reload_argument_string(self):
        args=train.reload_arguments(Path('.'),lambda **kw:kw)
        self.assertIsInstance(args['adapter_name_or_path'],str)
    def test_wrong_adapter_config(self):
        with self.assertRaises(ValueError): s.check_adapter({'r':16})
    def test_runtime_wrong_environment(self):
        with self.assertRaises(ValueError): s.diagnostic.environment_gate({'CUDA_VISIBLE_DEVICES':'0'})
        env,_=s.diagnostic.child_environment({'RANK':'0','WORLD_SIZE':'1','LOCAL_RANK':'0'})
        s.diagnostic.environment_gate(env); self.assertNotIn('RANK',env)
    def test_authorization_required(self):
        with self.assertRaises(ValueError): s.authorize_record('PLEASE IMPLEMENT THIS PLAN')
    def test_training_dataset_labels(self):
        records=[{'id':'test','total_tokens':2,'input_ids_sha256':s.digest([1,2]),'labels_sha256':s.digest([-100,2]),'assistant_turns':1}]
        self.assertTrue(s.check_dataset([{'input_ids':[1,2],'labels':[-100,2]}],records)['passed'])
        for data in ([],[{'input_ids':[1],'labels':[2]}],[{'input_ids':[1,2],'labels':[1,2]}]):
            with self.assertRaises(ValueError): s.check_dataset(data,records)
    def test_eval_unknown_source(self):
        with self.assertRaises(ValueError): ev.model_args('unapproved-adapter')
        with self.assertRaises(ValueError): ev.check_sequential({v:{'allocated_gpu':{'selection':'1','uuid':v,'name':'A100'},'started_at_utc':'2026-10-08T00:00:00+00:00','finished_at_utc':'2026-10-08T00:01:00+00:00'} for v in s.VARIANTS})
    def test_eval_run_provenance_mismatch(self):
        rows=[{'id':'case'}]; meta={'hash':'fixed'}; current={'source':'fixed'}; initial={'case':{'tokens':1}}
        run={'status':'complete','stage':'smoke','variant':s.VARIANTS[0],'provider_kind':'huggingface','versions':s.STACK,
            'policy':s.POLICY,'model_args':ev.model_args(s.VARIANTS[0]),'case_ids':['case'],'tasks_sha256':s.digest(rows),
            'frozen_dev':meta,'evidence':current,'adapter_sha256':s.START_SHA,'initial_prompt_audit':initial,
            'reference_context_audit_sha256':'reference','allocated_gpu':{'selection':'2','uuid':'GPU-test'},
            'semantic_review':'pending','human_review':'pending'}
        ev.check_run(run,'smoke',s.VARIANTS[0],rows,meta,current,initial,'reference')
        for key,value in {'tasks_sha256':'bad','adapter_sha256':'bad','evidence':{},'semantic_review':'human_approved',
            'initial_prompt_audit':{},'case_ids':[],'versions':{}}.items():
            with self.subTest(key=key),self.assertRaises(ValueError): ev.check_run({**run,key:value},'smoke',s.VARIANTS[0],rows,meta,current,initial,'reference')
    def test_missing_checkpoint_save_failure(self):
        with tempfile.TemporaryDirectory() as directory,self.assertRaises(ValueError): train.checkpoint_gate(Path(directory),'train')
    def test_stage_order_and_total_budget(self):
        from scripts.run_ticket_policy_retention_training_v1_2 import stage_sequence
        sequence=stage_sequence('all'); self.assertEqual(len(sequence),14)
        self.assertEqual([item[2] for item in sequence[:2]],['preflight','train'])
        self.assertEqual(sequence[-1][2],'targeted-comparison'); self.assertEqual(sum(s.COUNTS.values())*2,672)

class GPU2RevisionGuards(unittest.TestCase):
    def test_child_only_gpu2_no_distributed_residue(self):
        env,removed=s.child_environment({'CUDA_VISIBLE_DEVICES':'0,1','WORLD_SIZE':'1','RANK':'0','LOCAL_RANK':'0','MASTER_PORT':'29500','TORCHELASTIC_RUN_ID':'old'})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'2'); self.assertEqual(len(removed),5)
        s.environment_gate(env)
        for bad in ('0','1','0,1','0,1,2','2,1',''):
            with self.subTest(bad=bad),self.assertRaises(ValueError): s.environment_gate({**env,'CUDA_VISIBLE_DEVICES':bad})
        with self.assertRaises(ValueError): s.environment_gate({**env,'WORLD_SIZE':'1'})
    def test_legacy_gpu1_environment_unchanged(self):
        env,_=s.legacy_diagnostic.child_environment({})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'1'); s.legacy_diagnostic.environment_gate(env)
    def test_physical_gpu2_occupancy_only(self):
        identity='GPU-two, NVIDIA A100-SXM4-80GB, 81920, 81920\n'
        proof=s.check_gpu_rows(identity,'GPU-zero, 10, other\nGPU-one, 11, other\n')
        self.assertEqual(proof['selection'],'2')
        with self.assertRaises(ValueError): s.check_gpu_rows(identity,'GPU-two, 12, other\n')
        with self.assertRaises(ValueError): s.check_gpu_rows(identity.replace('81920, 81920','81920, 1000'),'')
        with self.assertRaises(ValueError): s.check_gpu_rows(identity.replace('GPU-two','MIG-two'),'')
    def test_cuda_uuid_physical_gpu2(self):
        s.check_actual_gpu_uuid('GPU-two','two')
        with self.assertRaises(ValueError): s.check_actual_gpu_uuid('GPU-two','GPU-one')
        with self.assertRaises(ValueError): s.check_actual_gpu_uuid('GPU-two',None)
    def test_runtime_queries_physical2_not_gpu1(self):
        from types import SimpleNamespace as N
        import os,sys
        env,_=s.child_environment({})
        torch=N(cuda=N(is_available=lambda:True,device_count=lambda:1,is_bf16_supported=lambda:True,
            get_device_properties=lambda logical:(self.assertEqual(logical,0) or N(uuid='two'))))
        replies=[N(stdout='GPU-two, NVIDIA A100-SXM4-80GB, 81920, 81920'),N(stdout='')]
        with patch.dict(os.environ,env,clear=True),patch.object(s.sys,'version_info',(3,11)),patch.dict(sys.modules,{'torch':torch}),patch.object(s.importlib.metadata,'version',side_effect=lambda k:s.STACK[k]),patch.object(s.subprocess,'run',side_effect=replies) as run:
            versions,gpu=s.runtime_gate()
            self.assertEqual(versions,s.STACK); self.assertEqual(gpu['selection'],'2')
            self.assertIn('--id=2',run.call_args_list[0].args[0]); self.assertNotIn('--id=1',run.call_args_list[0].args[0])
    def test_unchanged_recipe_and_frozen_selection(self):
        from extensions.ticket_policy_retention_training_v1_1 import support as prior
        for phase in ('train','preflight'):
            c=s.config_for(phase); p=prior.config_for(phase)
            for key in ('output_dir','dataset_dir','dataset'):
                c.pop(key); p.pop(key)
            self.assertEqual(c,p)
        self.assertEqual(s.inputs(check_selection=False)[3],prior.read(prior.REPORT/'preflight-selection.json'))
        self.assertEqual(s.RELEASE_SHA,prior.RELEASE_SHA); self.assertEqual(s.EVAL_SHA,prior.EVAL_SHA)
        self.assertEqual(s.START_SHA,prior.START_SHA)
        self.assertNotEqual(s.RUN,prior.RUN); self.assertEqual(s.GPU,'2')
    def test_prior_sources_and_reports_unchanged(self):
        self.assertEqual(s.sha(s.PARENT_DELIVERY),s.PARENT_DELIVERY_SHA)
        manifest=s.read(s.PARENT_DELIVERY)
        for group in ('sources','dependencies','report_files_sha256'): s.check_bindings(manifest[group])
    def test_gpu2_bound_authorization(self):
        with patch.object(s,'sha',return_value='delivery'):
            record=s.authorize_record('授权启动训练及配对评测')
        self.assertEqual(record['physical_gpu'],'2'); self.assertEqual(record['reserved_for_other_work'],['0','1'])
    def test_gpu1_eval_evidence_rejected(self):
        rows=[{'id':'case'}]; current={}; initial={}; reference=None
        run={'status':'complete','stage':'smoke','variant':s.VARIANTS[0],'provider_kind':'huggingface','versions':s.STACK,
            'policy':s.POLICY,'model_args':ev.model_args(s.VARIANTS[0]),'case_ids':['case'],'tasks_sha256':s.digest(rows),
            'frozen_dev':{},'evidence':current,'adapter_sha256':s.START_SHA,'initial_prompt_audit':initial,
            'reference_context_audit_sha256':reference,'allocated_gpu':{'selection':'1','uuid':'GPU-one'},
            'semantic_review':'pending','human_review':'pending'}
        with self.assertRaisesRegex(ValueError,'Wrong physical GPU'): ev.check_run(run,'smoke',s.VARIANTS[0],rows,{},current,initial,reference)
    def test_launcher_gpu2_for_all_stages(self):
        import inspect
        from scripts import run_ticket_policy_retention_training_v1_2 as launcher
        self.assertIn("'gpu': s.GPU",inspect.getsource(launcher.step))
        self.assertEqual(len(launcher.stage_sequence('all')),14)
        self.assertEqual(s.COUNTS,{'smoke':12,'dev':60,'diagnostic':72,'targeted':192})

class TargetedGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.rows=t.tasks()
    def task(self,variant): return next(r for r in self.rows if r['variant']==variant)
    def test_all_192_gold_and_replay(self):
        for task in self.rows:
            with self.subTest(id=task['id']):
                r=t.run_episode(task,t.GoldProvider(task)); self.assertTrue(r['scores']['task_success'],r['scores'])
                t.replay_record(task,r)
    def test_hidden_metadata_not_visible(self):
        task=self.rows[0]; p=t.input_payload([{'role':'user','content':task['request']}],t.system_prompt(task))
        self.assertEqual(set(p),{'messages','system','tools'})
        self.assertEqual(p['system'],task['initial_input']['system'])
        for key in ('fixture','goals','gold_episode_sha256','category','variant'): self.assertNotIn(key,p)
    def test_followup_reached_and_retained(self):
        task=self.task('email_followup'); r=t.run_episode(task,t.GoldProvider(task))
        self.assertTrue(r['followup_reached']); self.assertEqual(sum(m['role']=='user' for m in r['messages']),2)
        self.assertEqual(r['messages'][2]['content'],task['fixed_user_supplement'])
    def test_evidence_latest_version_state(self):
        b=t.Boundary('TABCDEFG01'); b.observe('get_ticket',{'ticket_id':'TABCDEFG01'},{'ok':True,'result':{'ticket_id':'TABCDEFG01','version':2,'status':'open'}})
        self.assertEqual(b.check('assign_ticket',{'ticket_id':'TABCDEFG01','assignee_id':'A002','expected_version':1}),'unobserved_version')
        self.assertEqual(b.check('update_ticket_status',{'ticket_id':'TABCDEFG01','expected_version':2,'status':'closed'}),'invalid_status_transition')
        self.assertEqual(b.check('create_ticket',{'requester_id':'UNOTKNOWN','title':'invented'}),'unobserved_argument')
    def test_conflict_reread_retry_limit(self):
        b=t.Boundary('TABCDEFG01'); args={'ticket_id':'TABCDEFG01','expected_version':2,'assignee_id':'A002'}
        b.tickets[args['ticket_id']]={'version':2,'status':'open'}; b.conflicts[args['ticket_id']]=False
        self.assertEqual(b.check('assign_ticket',args),'retry_without_reread')
        b.conflicts[args['ticket_id']]=True; b.retries[args['ticket_id']]=2
        self.assertEqual(b.check('assign_ticket',args),'retry_budget_exceeded')
    def test_partial_success(self):
        task=self.task('partial_assignment'); r=t.run_episode(task,t.GoldProvider(task))
        self.assertTrue(r['scores']['task_success']); self.assertIn('ticket_not_found',[e['response'].get('error',{}).get('code') for e in r['events']])
    def test_empty_success_not_fabricated(self):
        task=self.task('empty_kb'); r=t.run_episode(task,t.GoldProvider(task))
        self.assertEqual(r['events'][0]['response']['result'],[]); self.assertTrue(r['scores']['task_success'])
    def test_replay_tampering(self):
        task=self.task('lookup_only'); r=t.run_episode(task,t.GoldProvider(task)); r['final_text']='forged'
        with self.assertRaises(ValueError): t.replay_record(task,r)
    def test_failure_preserved_denominator(self):
        class Wrong:
            def prompt_info(self,p): return {'tokens':1}
            def generate(self,p): return {'text':'No action performed.','prompt_tokens':1,'finish_reason':'stop'}
        task=self.task('create_assign'); r=t.run_episode(task,Wrong()); result=t.summarize([r])
        self.assertEqual(result['generated_rows'],1); self.assertEqual(result['task_success'],{'correct':0,'total':1,'rate':0.0})
    def test_budget_exhausted_before_followup(self):
        task=self.task('email_followup')
        class Looper:
            def prompt_info(self,p): return {'tokens':1}
            def generate(self,p): return {'text':t.legacy_agent.render_calls([{'name':'get_ticket','arguments':{'ticket_id':'TUNKNOWN123'}}]),'prompt_tokens':1,'finish_reason':'stop'}
        r=t.run_episode(task,Looper()); self.assertEqual(len(r['rounds']),8); self.assertFalse(r['followup_reached']); self.assertFalse(r['scores']['task_success'])
    def test_executor_block_not_safe_model(self):
        task=self.task('create_denied')
        class Denied:
            def prompt_info(self,p): return {'tokens':1}
            def generate(self,p):
                text=t.legacy_agent.render_calls([{'name':'create_ticket','arguments':{'requester_id':'UMISSING123','title':'x','description':'y','priority':'normal'}}]) if len(p['messages'])==1 else 'Denied.'
                return {'text':text,'prompt_tokens':1,'finish_reason':'stop'}
        r=t.run_episode(task,Denied()); self.assertFalse(r['scores']['task_success']); self.assertGreater(r['scores']['constraint_violations'],0)
        self.assertEqual(r['events'][0]['response']['error']['code'],'write_not_authorized')
    def test_over_budget_denied_proposals_counted(self):
        task=self.task('create_denied')
        class Batch:
            def prompt_info(self,p): return {'tokens':1}
            def generate(self,p): return {'text':t.legacy_agent.render_calls([{'name':'create_ticket','arguments':{}}]*13),'prompt_tokens':1,'finish_reason':'stop'}
        r=t.run_episode(task,Batch())
        self.assertEqual(r['termination'],'call_budget_exceeded'); self.assertEqual(len(r['events']),0)
        result=t.summarize([r]); self.assertEqual(result['unauthorized_write_proposals'],13); self.assertGreater(result['constraint_violations'],0)
    def test_unreached_fault_model_failure_not_crash(self):
        task=self.task('conflict_exhaustion')
        class Bad:
            def prompt_info(self,p): return {'tokens':1}
            def generate(self,p):
                text=t.legacy_agent.render_calls([{'name':'unknown_tool','arguments':{}}]) if len(p['messages'])==1 else 'Not done.'
                return {'text':text,'prompt_tokens':1,'finish_reason':'stop'}
        r=t.run_episode(task,Bad()); self.assertFalse(r['scores']['task_success']); self.assertTrue(r['missed_faults'])
        t.replay_record(task,r)

if __name__=='__main__': unittest.main()
