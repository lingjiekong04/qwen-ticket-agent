"""Plan-only boundary tests; no simulations, tokenizer, model or GPU imports."""
import copy
import json
import unittest
from scripts import verify_ticket_policy_retention_plan_v1 as check


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((check.ROOT / check.PLAN).read_text(encoding='utf-8'))

    def reject(self, alter):
        plan = copy.deepcopy(self.plan)
        alter(plan)
        with self.assertRaises(ValueError):
            check.validate(plan)

    def test_current_plan_and_sources(self):
        self.assertTrue(check.inspect()['passed'])

    def test_no_training_authority(self):
        self.reject(lambda p: p['authorization'].update(training=True))

    def test_no_pilot_generated_claim(self):
        self.reject(lambda p: p['pilot'].update(status='complete'))

    def test_no_human_approval_inference(self):
        self.reject(lambda p: p['review'].update(human_approval='approved'))

    def test_dev_not_training(self):
        self.reject(lambda p: p['boundaries'].update(dev_in_training=True))

    def test_diagnostics_not_held_out(self):
        self.reject(lambda p: p['boundaries'].update(existing_diagnostics_are_held_out=True))

    def test_count(self):
        self.reject(lambda p: p['pilot']['cases'].pop())

    def test_evidence_cannot_refreeze(self):
        self.reject(lambda p: p['source_evidence'].update({next(iter(check.EVIDENCE)): '0'*64}))

    def test_policy_required(self):
        self.reject(lambda p: p['pilot']['cases'][0].update(allow_write=None))

    def test_denied_profiles_required(self):
        self.reject(lambda p: p['pilot']['cases'][6].update(permission_profile='sentence'))

    def test_no_training_payload(self):
        self.reject(lambda p: p['pilot']['cases'][0].update(events=[]))

    def test_counterfactual(self):
        self.reject(lambda p: p['pilot']['cases'][8].update(allow_write=False))


if __name__ == '__main__':
    unittest.main()
