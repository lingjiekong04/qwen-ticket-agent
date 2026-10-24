"""CPU-only specification gates; no inference, data construction or training."""
from collections import Counter, defaultdict
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import freeze_ticket_policy_retention_design_v1 as d


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.values=[d.read(d.OUT/name) for name in d.SOURCE_FILES[:4]]

    def invalid(self,index,mutate):
        values=deepcopy(self.values)
        mutate(values[index])
        with self.assertRaises(ValueError): d.validate(*values)

    def test_valid_specifications(self):
        d.validate(*self.values)

    def test_deterministic_96_metadata_pairs(self):
        cases=d.compile_cases(self.values[1])
        self.assertEqual(cases,d.compile_cases(deepcopy(self.values[1])))
        self.assertEqual(len(cases),96)
        d.validate_cases(cases,self.values[1])
        self.assertEqual(Counter(x['theme'] for x in cases),d.EVAL_QUOTAS)

    def test_no_instantiated_input_or_oracle(self):
        cases=d.compile_cases(self.values[1])
        for case in cases:
            for field in ('request','fixture','gold_answer','fault_schedule'):
                self.assertIsNone(case[field])
        cases[0]['request']='not authorized'
        with self.assertRaises(ValueError): d.validate_cases(cases,self.values[1])

    def test_nine_policy_only_contrasts(self):
        groups=defaultdict(list)
        for case in d.compile_cases(self.values[1]):
            if case['contrast_group']: groups[case['contrast_group']].append(case)
        self.assertEqual(len(groups),9)
        for rows in groups.values():
            self.assertEqual({x['allow_write'] for x in rows},{False,True})
            for field in ('profile','fixture_seed','business_family','surface_family'):
                self.assertEqual(rows[0][field],rows[1][field])

    def test_profiles_and_source_languages(self):
        cases=d.compile_cases(self.values[1])
        self.assertEqual(Counter(x['profile'] for x in cases),{p:32 for p in d.PROFILES})
        self.assertEqual(Counter(x['source_language'] for x in cases if x['theme']=='knowledge_chain'),{'zh':9,'en':9})

    def test_duplicate_blueprint_rejected(self):
        cases=d.compile_cases(self.values[1]); cases[1]=deepcopy(cases[0])
        with self.assertRaises(ValueError): d.validate_cases(cases,self.values[1])

    def test_acceptance_does_not_attest_human_inspection(self):
        self.invalid(0,lambda x:x.update(per_row_human_inspection_attested=True))

    def test_acceptance_source_hash_rejected(self):
        self.invalid(0,lambda x:x.update(pilot_episodes_sha256='0'*64))

    def test_acceptance_does_not_expand_authority(self):
        self.invalid(0,lambda x:x.update(bulk_generation_authorized=True))

    def test_seed_change_rejected(self):
        self.invalid(1,lambda x:x.update(seed=42))

    def test_eval_executable_authority_rejected(self):
        self.invalid(1,lambda x:x['authorization'].update(model_inference=True))

    def test_missing_variant_rejected_even_same_total(self):
        def mutate(x):
            variants=x['variant_pairs']['priority']
            variants['other']=variants.pop('handling_order')
        self.invalid(1,mutate)
        self.invalid(2,mutate)

    def test_family_overlap_rejected(self):
        self.invalid(1,lambda x:x['family_reservations'].update(training_business_families=x['family_reservations']['evaluation_business_families']))

    def test_legacy_dev_and_scores_preserved(self):
        self.invalid(1,lambda x:x['immutability'].update(legacy_dev_rows=59))
        self.invalid(1,lambda x:x['immutability'].update(legacy_scores_changed=True))

    def test_hidden_input_exclusion_required(self):
        self.invalid(1,lambda x:x['visible_inputs']['hidden_fields_excluded'].remove('fixture'))

    def test_retry_and_generation_budgets(self):
        self.invalid(1,lambda x:x['runtime_proposal'].update(conflict_retries_after_initial_attempt=3))
        self.invalid(1,lambda x:x['runtime_proposal'].update(max_generations_per_complete_episode=9))

    def test_batch_total_and_authority(self):
        self.invalid(2,lambda x:x.update(episodes=1802))
        self.invalid(2,lambda x:x['authorization'].update(extension_generation=True))

    def test_permission_pairs_balanced(self):
        def mutate(x):
            x['variant_pairs']['authority']['create_allowed']+=1
            x['variant_pairs']['authority']['create_denied']-=1
        self.invalid(2,mutate)

    def test_all_old_data_excluded(self):
        for field in ('old_data_mixed','accepted_pilot_copied_into_batch','dev_in_training','old_diagnostic_copied_into_training','old_final_test_read'):
            with self.subTest(field=field):
                self.invalid(2,lambda x:x['data_policy'].update({field:True}))

    def test_real_replay_and_failure_preservation(self):
        self.invalid(2,lambda x:x['data_policy'].update(real_simulated_returns=False))
        self.invalid(2,lambda x:x['data_policy'].update(failed_rows_dropped=True))

    def test_supervision_and_token_audit_not_bypassed(self):
        self.invalid(2,lambda x:x['supervision_proposal'].update(mask_history=True))
        self.invalid(2,lambda x:x['supervision_proposal'].update(real_token_label_audit='passed'))

    def test_training_recipe_not_confirmed(self):
        self.invalid(2,lambda x:x['experiment_proposal'].update(epochs=2))

    def test_paired_evaluation_budget(self):
        self.invalid(2,lambda x:x['experiment_proposal'].update(future_evaluation_episode_count=671))

    def test_extension_profiles_and_counterparts(self):
        self.invalid(3,lambda x:x['cases'][1].update(profile='json'))
        self.invalid(3,lambda x:x['cases'][1].update(variant='create_denied'))

    def test_existing_file_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='retention-design-test-') as temp:
            target=Path(temp)/'record.json'
            d.write_new(target,{'first':True})
            before=target.read_bytes()
            with self.assertRaises(FileExistsError): d.write_new(target,{'first':False})
            self.assertEqual(target.read_bytes(),before)

    def test_freeze_refuses_existing_output(self):
        with tempfile.TemporaryDirectory(prefix='retention-freeze-test-') as temp:
            output=Path(temp)
            d.write_new(output/'freeze.json',{'preserve':True})
            with patch.object(d,'OUT',output),patch.object(d,'gate',return_value=self.values):
                with self.assertRaisesRegex(ValueError,'refuse overwrite'): d.freeze()
            self.assertEqual(d.read(output/'freeze.json'),{'preserve':True})
            self.assertFalse((output/'evaluation-cases.json').exists())


if __name__=='__main__': unittest.main()
