"""No model imports; mutation tests for real replay and CPU evidence boundaries."""
import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from extensions.ticket_policy_retention_build_v1 import core as c,validate as v,pipeline as p,remote
from extensions.ticket_policy_audit_v1 import core as audit

class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eval_specs=c.specs('eval'); cls.train_specs=c.specs('train')
        cls.eval_episodes=[c.generate(s,l) for s in cls.eval_specs for l in ('zh','en')]
        cls.train_episodes=[c.generate(s,l) for s in cls.train_specs for l in ('zh','en')]

    def example(self,variant,locale='zh',split='train'):
        pool=self.train_episodes if split=='train' else self.eval_episodes
        return copy.deepcopy(next(e for e in pool if e['spec']['variant']==variant and e['locale']==locale))

    def rejects(self,e):
        expected=next(s for s in (self.train_specs if e['spec']['split']=='train' else self.eval_specs) if s['pair_id']==e['spec']['pair_id'])
        with self.assertRaises((ValueError,KeyError,IndexError)): v.validate_episode(e,expected)

    def test_counts_and_quotas(self):
        self.assertEqual(len(self.train_specs),900); self.assertEqual(len(self.eval_specs),96)
        self.assertEqual(Counter(s['theme'] for s in self.train_specs),{'knowledge_chain':240,'authority':180,'clarification':120,'faithful':120,'priority':120,'retention':120})

    def test_all_profiles_exact(self):
        for specs,n in ((self.train_specs,300),(self.eval_specs,32)):
            self.assertEqual(Counter(s['permission_profile'] for s in specs),{p:n for p in c.PROFILES})

    def test_all_public_episodes_replay(self):
        byid={s['pair_id']:s for s in self.eval_specs}
        for e in self.eval_episodes: self.assertTrue(v.validate_episode(e,byid[e['spec']['pair_id']])['passed'])

    def test_training_variants_replay(self):
        variants={e['spec']['variant']:e for e in self.train_episodes}
        for e in variants.values(): self.assertTrue(v.validate_episode(e,e['spec'])['passed'])

    def test_counterfactual_initial_and_request_equal(self):
        for specs in (self.train_specs,self.eval_specs):
            groups={s['contrast_group'] for s in specs if s['contrast_group']}
            for group in groups:
                a,b=[s for s in specs if s['contrast_group']==group]
                self.assertNotEqual(a['allow_write'],b['allow_write'])
                self.assertEqual(a['fixture'],b['fixture'])
                for l in ('zh','en'): self.assertEqual(c.request(a,l),c.request(b,l))

    def test_denied_zero_calls_even_missing_and_override(self):
        for e in self.eval_episodes:
            if e['spec']['variant'].endswith('_denied'):
                self.assertFalse(e['events']); self.assertEqual(e['claims'][0]['claims'][0]['kind'],'denied')

    def test_permission_missing(self):
        e=self.example('create_allowed'); e['row']['system']=e['row']['system'].split('\n')[0]; self.rejects(e)

    def test_permission_conflict(self):
        e=self.example('create_denied'); e['row']['system']+='\nWrites allowed'; self.rejects(e)

    def test_private_metadata_leak(self):
        e=self.example('create_allowed'); e['row']['fixture']=e['initial_state']; self.rejects(e)

    def test_invalid_schema(self):
        e=self.example('assign_allowed'); call=json.loads(e['row']['conversations'][-3]['value']); call[0]['arguments']['permission']=True
        e['row']['conversations'][-3]['value']=c.canonical(call); self.rejects(e)

    def test_unobserved_version(self):
        e=self.example('assign_allowed'); call=json.loads(e['row']['conversations'][-3]['value']); call[0]['arguments']['expected_version']+=1
        e['row']['conversations'][-3]['value']=c.canonical(call); self.rejects(e)

    def test_unobserved_requester(self):
        e=self.example('created_unassigned'); call=json.loads(e['row']['conversations'][-3]['value']); call[0]['arguments']['requester_id']='UNOOBSERVED123'
        e['row']['conversations'][-3]['value']=c.canonical(call); self.rejects(e)

    def test_fabricated_success(self):
        e=self.example('created_unassigned'); e['events'][-1]['response']['result']['assignee_id']='A002'; self.rejects(e)

    def test_wrong_final_assignment_claim(self):
        e=self.example('created_unassigned'); e['claims'][-1]['claims'][0]['ticket']['assignee_id']='A001'; self.rejects(e)

    def test_missing_assistant_statement(self):
        e=self.example('email_followup'); e['claims'].pop(0); self.rejects(e)

    def test_lost_followup(self):
        e=self.example('description_followup'); del e['row']['conversations'][1:3]; self.rejects(e)

    def test_supplied_fields_preserved(self):
        for variant in ('email_followup','title_followup','description_followup','priority_followup'):
            e=self.example(variant); e['events'][-1]['call']['arguments']['title']='Guessed title'; self.rejects(e)

    def test_missing_fields_are_not_in_context(self):
        for s in self.eval_specs+self.train_specs:
            m=s['missing_field']
            if not m: continue
            for l in ('zh','en'):
                text=c.request(s,l)
                if m=='priority': self.assertNotIn('"priority":',text)
                else: self.assertNotIn(s['fixture']['tickets'][0]['ticket_id'] if m=='ticket_id' else s['fields'][m],text)

    def test_known_closed_write_rejected(self):
        e=self.example('partial_assignment'); e['events'][2]['response']['result']['status']='closed'; self.rejects(e)

    def test_partial_failure_real_delete_witness(self):
        e=self.example('partial_assignment')
        self.assertEqual(e['events'][2]['response']['result']['status'],'resolved')
        self.assertEqual(e['external_events'][0]['kind'],'ticket_delete')
        self.assertEqual(len(e['final_state']['tickets']),1)
        self.assertEqual(e['events'][3]['response']['error']['code'],'ticket_not_found')

    def test_lost_external_witness(self):
        e=self.example('partial_assignment'); e['external_events']=[]; self.rejects(e)

    def test_conflict_success_one_and_two(self):
        counts=set()
        for e in self.eval_episodes:
            if e['spec']['variant']=='conflict_success':
                counts.add(len(e['external_events'])); self.assertTrue(e['events'][-1]['response']['ok'])
        self.assertEqual(counts,{1,2})

    def test_conflict_exhaustion_three_no_success(self):
        e=self.example('conflict_exhaustion'); names,errors=c.expected_chain(e['spec'])
        self.assertEqual(names,['get_ticket','assign_ticket']*3); self.assertEqual(errors,['version_conflict']*3)
        self.assertEqual(e['claims'][-1]['claims'][0]['kind'],'conflict')

    def test_conflict_retry_missing_read(self):
        e=self.example('conflict_success'); del e['row']['conversations'][5:7]; self.rejects(e)

    def test_skipped_status(self):
        e=self.example('status_sequence'); call=json.loads(e['row']['conversations'][3]['value']); call[0]['arguments']['status']='closed'
        e['row']['conversations'][3]['value']=c.canonical(call); self.rejects(e)

    def test_dual_read_all_combinations(self):
        combos={tuple(sorted(x['priority'] for x in s['fixture']['tickets'])) for s in self.eval_specs if s['variant']=='read_two'}
        self.assertEqual(combos,{('low','normal'),('high','low'),('high','normal'),('normal','normal')})

    def test_source_language_balance(self):
        for specs,n in ((self.eval_specs,9),(self.train_specs,120)):
            self.assertEqual(Counter(s['source_language'] for s in specs if s['theme']=='knowledge_chain'),{'zh':n,'en':n})

    def test_titles_only_stops(self):
        e=self.example('titles_only'); self.assertEqual([x['call']['name'] for x in e['events']],['search_kb'])

    def test_omitted_body_detected(self):
        e=self.example('multiple_bodies'); e['claims'][-1]['claims'].pop(); self.rejects(e)

    def test_source_fact_hallucination(self):
        e=self.example('single_body'); e['claims'][-1]['claims'][0]['facts'][0]['en']='Delete every original'; self.rejects(e)

    def test_priority_answer_does_not_reverse_order(self):
        e=self.example('handling_order'); self.assertIn('high → normal → low',e['row']['conversations'][-1]['value'])
        self.assertNotIn('high → normal → low',e['row']['conversations'][0]['value'])
        e['claims'][-1]['claims'][0]['order']=['low','normal','high']; self.rejects(e)

    def test_wrong_sla_detected(self):
        e=self.example('no_sla'); e['claims'][-1]['claims'][0]['sla']='2 hours'; self.rejects(e)

    def test_no_old_eval_gold_read_by_train_factory(self):
        original=c.read
        def checked(path):
            self.assertFalse(Path(path).is_relative_to(c.EVAL)); return original(path)
        with patch.object(c,'read',side_effect=checked): self.assertEqual(len(c.specs('train')),900)

    def test_new_split_isolation(self):
        result=v.isolation(self.train_episodes,self.eval_episodes,{'ids':set(),'normalized_requests':set()})
        self.assertFalse(any(result['eval_training_overlap_counts'].values()))
        self.assertEqual(result['training_pair_count'],900)

    def test_copied_targeted_instance_rejected(self):
        with self.assertRaises(ValueError): v.isolation(self.eval_episodes,self.eval_episodes,{'ids':set(),'normalized_requests':set()})

    def test_new_eval_oracle_has_complete_visible_input(self):
        e=self.example('priority_followup',split='eval'); task=p.oracle(e)
        self.assertEqual(task['initial_input']['messages'],[e['row']['conversations'][0]])
        self.assertIsNotNone(task['fixed_user_supplement']); self.assertFalse(task['model_execution_authorized'])
        self.assertNotIn('fixture',task['initial_input']); self.assertIn('fixture',task)

    def test_sample_fixed_60_pairs_and_risks(self):
        lengths={e['id']:1000+len(c.canonical(e['row']))%1000 for e in self.train_episodes}
        sample,meta=remote.select_sample(self.train_episodes,lengths)
        self.assertEqual(len(sample),120); self.assertEqual(meta['allocation'],remote.SAMPLE)
        self.assertEqual(remote.select_sample(self.train_episodes,lengths),(sample,meta))
        self.assertEqual([e['locale'] for e in sample],['zh','en']*60)

    def test_sample_needs_all_real_lengths(self):
        with self.assertRaises(ValueError): remote.select_sample(self.train_episodes,{})

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'existing.json'; p.write(path,{'preserved':True})
            with self.assertRaises(ValueError): p.fresh(path)
            with self.assertRaises(FileExistsError): p.write(path,{})
            self.assertEqual(c.read(path),{'preserved':True})

    def test_mask_all_assistant_targets(self):
        result=audit.verify_labels([([1,2],[3]),([4,5],[6,7])],[1,2,3,4,5,6,7],[-100,-100,3,-100,-100,6,7],2)
        self.assertEqual(result['assistant_turns'],2)

    def test_user_or_tool_label_leak_rejected(self):
        with self.assertRaises(ValueError): audit.verify_labels([([1,2],[3])],[1,2,3],[1,-100,3],1)

    def test_missing_assistant_labels_rejected(self):
        with self.assertRaises(ValueError): audit.verify_labels([([1,2],[3])],[1,2,3],[-100,-100,-100],1)

    def test_truncated_tokens_rejected(self):
        with self.assertRaises(ValueError): audit.verify_labels([([1,2],[3])],[1,2],[-100,-100],1)

    def test_dropped_turn_rejected(self):
        with self.assertRaises(ValueError): audit.verify_labels([([1,2],[3])],[1,2,3],[-100,-100,3],2)

    def test_converter_role_change_rejected(self):
        row=self.example('created_unassigned')['row']
        converted={'_prompt':[{'role':'user','content':m['value']} for m in row['conversations'][:-1]],'_response':[], '_system':row['system'],'_tools':row['tools']}
        with self.assertRaises(ValueError): audit.check_converted(row,converted)

    def test_converter_tool_content_change_rejected(self):
        row=self.example('created_unassigned')['row']
        msgs=[{'role':audit.ROLES[m['from']],'content':m['value']} for m in row['conversations']]
        msgs[1]['content']='[]'
        with self.assertRaises(ValueError): audit.check_converted(row,{'_prompt':msgs[:-1],'_response':msgs[-1:],'_system':row['system'],'_tools':row['tools']})

    def test_offline_cpu_environment(self):
        env,removed=audit.cpu_environment({'RANK':'0','WORLD_SIZE':'1','CUDA_VISIBLE_DEVICES':'1'})
        audit.cpu_gate(env); self.assertEqual(env['CUDA_VISIBLE_DEVICES'],''); self.assertIn('RANK',removed)

    def test_gpu_visible_rejected(self):
        env,_=audit.cpu_environment({}); env['CUDA_VISIBLE_DEVICES']='1'
        with self.assertRaises(ValueError): audit.cpu_gate(env)

    def test_network_loading_rejected(self):
        env,_=audit.cpu_environment({}); env['HF_HUB_OFFLINE']='0'
        with self.assertRaises(ValueError): audit.cpu_gate(env)

    def test_supervision_fixed(self):
        self.assertEqual(p.SUPERVISION,{'template':'qwen','cutoff_len':4096,'train_on_prompt':False,'mask_history':False,'packing':False})

    def test_source_gate_rejects_changed_design(self):
        with patch.object(c,'sha',return_value='bad'):
            with self.assertRaises(ValueError): c.specs('train')

    def test_authorization_and_missing_evidence(self):
        with patch.object(c,'read',return_value={}):
            with self.assertRaises((ValueError,KeyError)): p.acceptance_gate()

if __name__=='__main__': unittest.main()
