from __future__ import annotations

import unittest

from . import core


class DesignTests(unittest.TestCase):
    def test_all_deterministic_cases(self):
        cases = core.validation_cases()
        self.assertEqual(len(cases), 13)
        self.assertTrue(all(item["passed"] for item in cases))

    def test_read_only_blocks_write(self):
        scope = {"allowed_tools": ["get_ticket"], "target_ticket_ids": ["T1"], "allowed_mutations": {}}
        result = core.preflight(scope, {"name": "assign_ticket", "arguments": {"ticket_id": "T1", "assignee_id": "A1"}})
        self.assertFalse(result["allowed"])
        self.assertIn("operation_outside_request_scope", result["reasons"])

    def test_unrelated_state_change_is_rejected(self):
        scope = {"allowed_tools": ["update_ticket_status"], "target_ticket_ids": ["T1"],
                 "allowed_mutations": {"update_ticket_status": ["status"]}}
        result = core.state_diff(scope, {"ticket_id": "T1", "status": "open", "assignee_id": None},
                                 {"ticket_id": "T1", "status": "in_progress", "assignee_id": "A1"})
        self.assertFalse(result["accepted"])

    def test_family_failure_cannot_be_averaged_away(self):
        result = core.assess_admission({"forced_recovered": 10}, {
            "forced_recovered": 12, "invalid_outputs": 0,
            "forced_families": {"illegal_transition": [0, 2], "other": [12, 12]},
        })
        self.assertFalse(result["model_recovery_expansion_eligible"])

    def test_safety_and_recovery_are_separate(self):
        result = core.assess_admission({"forced_recovered": 10}, {
            "forced_recovered": 10, "invalid_outputs": 0,
            "forced_families": {"illegal_transition": [0, 2]},
        })
        self.assertTrue(result["safety_controller_eligible"])
        self.assertFalse(result["model_recovery_expansion_eligible"])


if __name__ == "__main__":
    unittest.main()
