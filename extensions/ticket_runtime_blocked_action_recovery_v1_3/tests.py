from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from extensions.ticket_runtime_contract_guard_v1_1 import core as guard
from extensions.ticket_runtime_contract_guard_integration_v1 import core as runtime_v1
from . import core


ROOT = Path(__file__).resolve().parents[2]
TASKS = json.loads((ROOT / "data/ticket_runtime_full_workflow_dev_v2_1/tasks.json").read_text(encoding="utf-8"))


def task(family: str, entry: str = "forced_hazard_prefix") -> dict:
    return copy.deepcopy(next(item for item in TASKS
                              if item["family"] == family and item["entry_kind"] == entry))


class BlockedActionRecoveryTests(unittest.TestCase):
    def test_01_wrong_assignee_emits_exact_call(self):
        episode = core.BlockedActionRecoveryEpisode(task("assign_ticket"))
        runtime_v1.seed_prefix(episode, [])
        contract = episode.events[-1]["response"]["error"]["recovery_contract"]
        self.assertEqual(contract["trigger_family"], "wrong_assignee")
        self.assertEqual(contract["required_next_call"]["arguments"]["assignee_id"],
                         episode.workflow.target_assignee_id)

    def test_02_assign_exact_recovery_completes(self):
        item = task("assign_ticket")
        state = item["initial_state"]["tickets"][0]
        provider = core.ScriptedProvider([guard.wrapped(
            "assign_ticket", ticket_id=state["ticket_id"],
            assignee_id=item["target_assignee_id"], expected_version=state["version"])])
        record = core.run_episode(item, provider, variant="synthetic")
        self.assertEqual(record["termination"], "final_response")
        self.assertEqual(record["after"][state["ticket_id"]]["assignee_id"],
                         item["target_assignee_id"])

    def test_03_lookup_exact_recovery_completes(self):
        item = task("lookup_user")
        provider = core.ScriptedProvider([guard.wrapped("lookup_user", email=item["trusted_email"])])
        record = core.run_episode(item, provider, variant="synthetic")
        self.assertEqual(record["termination"], "final_response")

    def test_04_search_exact_recovery_completes(self):
        item = task("search_tickets")
        provider = core.ScriptedProvider([guard.wrapped("search_tickets", **item["trusted_search_filters"])])
        record = core.run_episode(item, provider, variant="synthetic")
        self.assertEqual(record["termination"], "final_response")

    def test_05_repeated_wrong_recovery_never_reaches_backend(self):
        item = task("lookup_user")
        wrong = item["fault_plan"]["call"]
        provider = core.ScriptedProvider([
            guard.wrapped(wrong["name"], **wrong["arguments"]),
            guard.wrapped(wrong["name"], **wrong["arguments"]),
        ])
        record = core.run_episode(item, provider, variant="synthetic")
        self.assertEqual(record["termination"], "safe_stop")
        model_events = [event for event in record["events"] if event["provenance"] == "model_proposal"]
        self.assertTrue(model_events)
        self.assertFalse(any(event["reached_backend"] for event in model_events))

    def test_06_plain_text_nonprogress_safely_stops(self):
        item = task("search_tickets")
        record = core.run_episode(item, core.ScriptedProvider(["Cannot.", "Still cannot."]),
                                  variant="synthetic")
        self.assertEqual(record["termination"], "safe_stop")
        self.assertFalse(json.loads(record["final_text"])["operation_completed"])

    def test_07_direct_lookup_path_is_unchanged(self):
        item = task("lookup_user", "direct_control")
        record = core.run_episode(item, core.ScriptedProvider([
            guard.wrapped("lookup_user", email=item["trusted_email"])]), variant="synthetic")
        self.assertEqual(record["termination"], "final_response")
        self.assertIsNone(record["recovery"])

    def test_08_direct_search_path_is_unchanged(self):
        item = task("search_tickets", "direct_control")
        record = core.run_episode(item, core.ScriptedProvider([
            guard.wrapped("search_tickets", **item["trusted_search_filters"])]), variant="synthetic")
        self.assertEqual(record["termination"], "final_response")
        self.assertIsNone(record["recovery"])

    def test_09_direct_assign_path_is_unchanged(self):
        item = task("assign_ticket", "direct_control")
        state = item["initial_state"]["tickets"][0]
        record = core.run_episode(item, core.ScriptedProvider([
            guard.wrapped("get_ticket", ticket_id=state["ticket_id"]),
            guard.wrapped("assign_ticket", ticket_id=state["ticket_id"],
                          assignee_id=item["target_assignee_id"], expected_version=state["version"]),
        ]), variant="synthetic")
        self.assertEqual(record["termination"], "final_response")
        self.assertIsNone(record["recovery"])

    def test_10_frozen_guard_hash_is_unchanged(self):
        import hashlib
        path = ROOT / "extensions/ticket_runtime_contract_guard_v1_1/core.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "db9630c246662baea0b3a81c342475142f88ddb0c4971681e8f61ba89c0f4260")


if __name__ == "__main__":
    unittest.main()

