"""CPU correctness and rejection tests; not a model quality or tokenizer audit."""
import copy
import unittest
from unittest.mock import patch
from extensions.ticket_policy_retention_pilot_v1 import core as c, pipeline as p, validate as v


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = c.specs()
        cls.episodes = [c.generate(s,l) for s in cls.specs for l in ('zh','en')]

    def case(self,n,locale='zh'):
        return copy.deepcopy(self.episodes[(n-1)*2+(locale=='en')])

    def reject(self,e):
        with self.assertRaises((ValueError,KeyError,IndexError)):
            v.validate_episode(e)

    def test_all_rows_independent_replay(self):
        self.assertEqual(v.validate_population(self.episodes)['rows'],40)

    def test_counterfactual(self):
        a,b = self.case(6),self.case(9)
        self.assertEqual(a['initial_state'],b['initial_state'])
        self.assertEqual(a['row']['conversations'][0],b['row']['conversations'][0])
        self.assertFalse(a['events'])
        self.assertEqual(len(b['events']),2)

    def test_bilingual_consistency(self):
        e = copy.deepcopy(self.episodes)
        e[1]['final_state']['articles'].clear()
        with self.assertRaises(ValueError):
            v.validate_population(e)

    def test_missing_permission(self):
        e = self.case(7)
        e['row']['system']='Support assistant.'
        self.reject(e)

    def test_write_proposal_not_excused_by_block(self):
        e = self.case(7)
        call = {'name':'assign_ticket','arguments':{'ticket_id':e['spec']['fixture']['tickets'][0]['ticket_id'],
                'assignee_id':'A002','expected_version':e['spec']['fixture']['tickets'][0]['version']}}
        final = e['row']['conversations'][1]
        e['row']['conversations'][1:] = [{'from':'function_call','value':c.canonical([call])},
            {'from':'observation','value':c.canonical([{'ok':False,'error':{'code':'write_not_authorized'}}])},final]
        e['claims'][0]['index']=3
        self.reject(e)

    def test_metadata_leak(self):
        e = self.case(1)
        e['row']['fixture']=e['initial_state']
        self.reject(e)

    def test_premature_stop(self):
        e = self.case(1)
        del e['row']['conversations'][3:5]
        self.reject(e)

    def test_title_only_no_body(self):
        self.assertEqual([x['call']['name'] for x in self.case(3)['events']],['search_kb'])

    def test_real_partial_body_error(self):
        e = self.case(5)
        self.assertEqual(len(e['events'][0]['response']['result']),2)
        self.assertEqual(len(e['events'][0]['after']['articles']),1)
        self.assertEqual(e['events'][2]['response']['error']['code'],'article_not_found')
        self.assertEqual(e['external_events'][0]['kind'],'external_article_delete')

    def test_fabricated_observation(self):
        e = self.case(1)
        e['events'][1]['response']['result']['content']='Invented body'
        self.reject(e)

    def test_no_argument_guess(self):
        e = self.case(9)
        e['row']['conversations'][1]['value']=c.canonical([{'name':'lookup_user','arguments':{'email':'guess@example.test'}}])
        self.reject(e)

    def test_wrong_role(self):
        e = self.case(1)
        e['row']['conversations'][2]['from']='gpt'
        self.reject(e)

    def test_followup_fields(self):
        for n in (14,15):
            e = self.case(n)
            r = v.validate_episode(e)
            self.assertEqual(r['assistant_message_indices_expected_supervised'],[1,3,5,7])
            self.assertEqual(r['message_indices_expected_masked'],[0,2,4,6])
            e['row']['conversations'][2]['value']='Please continue.'
            self.reject(e)

    def test_unobserved_version(self):
        e = self.case(16)
        call = copy.deepcopy(e['events'][2]['call'])
        call['arguments']['expected_version']=1999
        e['row']['conversations'][5]['value']=c.canonical([call])
        self.reject(e)

    def test_status_sequence(self):
        e = self.case(17)
        self.assertEqual([x['call']['arguments'].get('status') for x in e['events'][1:]],['in_progress','resolved','closed'])
        call = copy.deepcopy(e['events'][1]['call'])
        call['arguments']['status']='closed'
        e['row']['conversations'][3]['value']=c.canonical([call])
        self.reject(e)

    def test_conflict_witness_and_bound(self):
        e = self.case(18)
        self.assertEqual(len(e['events']),6)
        self.assertEqual(len(e['external_events']),3)
        self.assertTrue(all(not e['events'][i]['response']['ok'] for i in (1,3,5)))
        e['external_events'][0]['new_version']+=1
        self.reject(e)

    def test_priority_order(self):
        e = self.case(11,'en')
        e['claims'][0]['claims'][0]['order']=['low','normal','high']
        self.reject(e)

    def test_sla_claim(self):
        e = self.case(12)
        e['claims'][0]['claims'][0]['sla']='2 hours'
        self.reject(e)

    def test_not_found_stops(self):
        e = self.case(20)
        self.assertEqual([x['call']['name'] for x in e['events']],['lookup_user'])
        self.assertEqual(e['initial_state'],e['final_state'])

    def test_existing_output_refused(self):
        with patch.object(p.c,'OUT',c.HERE):
            with self.assertRaises(ValueError):
                p.fresh()

    def test_plan_hash_not_refrozen(self):
        with patch.object(c,'PLAN_SHA','0'*64):
            with self.assertRaises(ValueError):
                c.specs()


if __name__=='__main__':
    unittest.main()
