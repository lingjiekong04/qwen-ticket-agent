import copy
import unittest
from unittest.mock import patch
from extensions.ticket_policy_retention_extension_v1_1 import core as c,validate as v,pipeline as p


class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.episodes=[c.generate(s,l) for s in c.specs() for l in ('zh','en')]

    def case(self,n,locale='zh'): return copy.deepcopy(self.episodes[(n-1)*2+(locale=='en')])

    def reject(self,e):
        with self.assertRaises((ValueError,KeyError,IndexError)): v.validate_episode(e)

    def test_all_rows_replay(self): self.assertEqual(v.validate_population(self.episodes)['rows'],24)

    def test_all_three_permission_counterfactuals(self):
        for a,b in ((1,2),(3,4),(5,6)):
            denied,allowed=self.case(a),self.case(b)
            self.assertEqual(denied['initial_state'],allowed['initial_state'])
            self.assertEqual(denied['row']['conversations'][0],allowed['row']['conversations'][0])
            self.assertFalse(denied['events']); self.assertTrue(allowed['events'])
            self.assertNotEqual(denied['row']['system'],allowed['row']['system'])

    def test_no_call_permission_refusal_not_executor_credit(self):
        e=self.case(3); e['row']['conversations'][1]={'from':'function_call','value':c.canonical([{'name':'get_ticket','arguments':{'ticket_id':e['spec']['fixture']['tickets'][0]['ticket_id']}}])}
        self.reject(e)

    def test_permission_field_missing(self):
        e=self.case(1); e['row']['system']='Support assistant'; self.reject(e)

    def test_schema_extra_permission(self):
        e=self.case(2); call=copy.deepcopy(e['events'][0]['call']); call['arguments']['allow_write']=True
        e['row']['conversations'][1]['value']=c.canonical([call]); self.reject(e)

    def test_hidden_metadata(self):
        e=self.case(7); e['row']['fixture']=e['initial_state']; self.reject(e)

    def test_real_chinese_body_paraphrases(self):
        zh,en=self.case(7),self.case(7,'en')
        self.assertEqual(zh['events'],en['events'])
        self.assertEqual(zh['events'][1]['response']['result']['content'],c.BODY)
        self.assertIn('先记下',zh['row']['conversations'][-1]['value'])
        self.assertIn('First note',en['row']['conversations'][-1]['value'])

    def test_unobserved_article_body(self):
        e=self.case(7); e['claims'][0]['claims'][0]['article']['content']='invented'; self.reject(e)

    def test_summary_fact_alignment(self):
        e=self.case(7); e['claims'][0]['claims'][0]['facts'][0]['zh']='先删除原始记录'; self.reject(e)

    def test_knowledge_chain_stops_after_body(self):
        self.assertEqual([x['call']['name'] for x in self.case(7)['events']],['search_kb','get_kb_article'])
        e=self.case(7); del e['row']['conversations'][3:5]; self.reject(e)

    def test_two_conflicts_then_success(self):
        e=self.case(8)
        self.assertEqual([x['response'].get('error',{}).get('code') for x in e['events'][1::2]],['version_conflict','version_conflict',None])
        self.assertEqual(len(e['external_events']),2)
        self.assertEqual(e['events'][-1]['response']['result']['assignee_id'],'A002')
        self.assertEqual(len({x['key'] for x in e['events'] if x['key']}),3)

    def test_conflict_witness(self):
        e=self.case(8); e['external_events'][0]['new_version']+=1; self.reject(e)

    def test_unobserved_version(self):
        e=self.case(8); call=copy.deepcopy(e['events'][1]['call']); call['arguments']['expected_version']+=10
        e['row']['conversations'][3]['value']=c.canonical([call]); self.reject(e)

    def test_fabricated_success(self):
        e=self.case(8); e['events'][1]['response']={'ok':True,'result':e['events'][-1]['response']['result']}; self.reject(e)

    def test_read_two_different_priority(self):
        e=self.case(9)
        self.assertEqual([x['response']['result']['priority'] for x in e['events']],['low','high'])
        self.assertEqual(e['initial_state'],e['final_state'])
        e['claims'][0]['claims'][1]['ticket']['priority']='low'; self.reject(e)

    def test_continuation_and_supervision_structure(self):
        for n in (10,11):
            e=self.case(n); result=v.validate_episode(e)
            self.assertEqual(result['assistant_message_indices_expected_supervised'],[1,3,5,7])
            self.assertEqual(result['message_indices_expected_masked'],[0,2,4,6])
            self.assertEqual(e['events'][1]['response']['result']['description'],e['spec']['fields']['description'])
            self.assertEqual(e['events'][1]['response']['result']['priority'],e['spec']['fields']['priority'])
            e['row']['conversations'][2]['value']='Continue'; self.reject(e)

    def test_continuation_description_not_guessed(self):
        e=self.case(10); e['row']['conversations'][1]['value']='I will choose a description.'; self.reject(e)

    def test_created_unassigned(self):
        for n in (2,10,11): self.assertIsNone(self.case(n)['events'][-1]['response']['result']['assignee_id'])

    def test_partial_success_actual_post_read_absence(self):
        e=self.case(12)
        self.assertEqual(e['events'][1]['response']['result']['assignee_id'],'A002')
        self.assertEqual(e['events'][2]['response']['result']['status'],'resolved')
        self.assertEqual(e['events'][3]['response']['error']['code'],'ticket_not_found')
        self.assertEqual(len(e['final_state']['tickets']),1)
        self.assertEqual(e['external_events'][0]['after_call'],3)
        self.assertEqual(e['external_events'][0]['kind'],'external_ticket_delete')
        e['claims'][0]['claims'].pop(); self.reject(e)

    def test_known_closed_target_rejected_before_tool(self):
        with self.assertRaisesRegex(ValueError,'Known closed'):
            v.assignment_target_gate({'tickets':{'known':{'status':'closed'}}},{'ticket_id':'known'})

    def test_partial_delete_witness_required(self):
        e=self.case(12); e['external_events']=[]; self.reject(e)

    def test_partial_delete_not_before_read(self):
        e=self.case(12); e['external_events'][0]['after_call']=2; self.reject(e)

    def test_final_does_not_claim_hidden_deletion_cause(self):
        for locale in ('zh','en'):
            text=self.case(12,locale)['row']['conversations'][-1]['value']
            self.assertNotIn('外部删除',text); self.assertNotIn('external deletion',text)
            self.assertIn('ticket_not_found',text)

    def test_only_affected_pair_changed(self):
        result=p.revision_comparison(self.episodes)
        self.assertEqual(len(result['unchanged_rows']),22)
        self.assertEqual([x['id'] for x in result['changed_rows']],['nx1-p12-zh','nx1-p12-en'])

    def test_unaffected_answer_change_rejected(self):
        episodes=copy.deepcopy(self.episodes); episodes[0]['row']['conversations'][-1]['value']+=' altered'
        with self.assertRaises(ValueError): p.revision_comparison(episodes)

    def test_bilingual_consistency(self):
        episodes=copy.deepcopy(self.episodes); episodes[1]['initial_state']['tickets'].clear()
        with self.assertRaises(ValueError): v.validate_population(episodes)

    def test_role_order(self):
        e=self.case(2); e['row']['conversations'][2]['from']='gpt'; self.reject(e)

    def test_existing_outputs_refused(self):
        with patch.object(c,'OUT',c.HERE):
            with self.assertRaises(ValueError): p.fresh()

    def test_design_hash_not_refrozen(self):
        with patch.object(c,'FREEZE_SHA','0'*64):
            with self.assertRaises(ValueError): c.specs()


if __name__=='__main__': unittest.main()
