from __future__ import annotations

import copy
import unittest

from extensions.ticket_runtime_contract_guard_integration_v1 import core as runtime_v1
from extensions.ticket_runtime_trusted_completion_v1_7 import core as c
from extensions.ticket_runtime_trusted_completion_dev_v1 import core as population


class TrustedCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks, _ = population.generate()

    def task(self, group: str) -> dict:
        return copy.deepcopy(next(task for task in self.tasks if task["test_group"] == group))

    def test_01_search_fault_is_blocked_then_exactly_recovered(self):
        task = self.task("exact_search_recovery")
        row = c.run_episode(task, runtime_v1.ScriptedProvider([]), variant="synthetic")
        self.assertEqual(row["termination"], "final_response")
        self.assertEqual(len(row["rounds"]), 0)
        self.assertFalse(row["events"][0]["reached_backend"])
        self.assertTrue(any(event["provenance"] == "guard_recovery" and
                            event["call"]["name"] == "search_tickets" and
                            event["reached_backend"] for event in row["events"]))

    def test_02_case_mutation_never_reaches_backend(self):
        task = next(copy.deepcopy(task) for task in self.tasks
                    if task["test_group"] == "exact_search_recovery" and
                    task["fault_plan"]["kind"] == "identifier_case_mutation")
        row = c.run_episode(task, runtime_v1.ScriptedProvider([]), variant="synthetic")
        self.assertFalse(row["events"][0]["reached_backend"])
        self.assertEqual(row["events"][-1]["call"]["arguments"], task["trusted_search_filters"])

    def test_03_wrong_ticket_update_rereads_and_updates_only_target(self):
        task = self.task("trusted_target_reread_recovery")
        row = c.run_episode(task, runtime_v1.ScriptedProvider([]), variant="synthetic")
        target = task["initial_state"]["tickets"][0]["ticket_id"]
        self.assertEqual([event["call"]["name"] for event in row["events"][-2:]],
                         ["get_ticket", "update_ticket_status"])
        self.assertTrue(all(event["call"]["arguments"]["ticket_id"] == target
                            for event in row["events"][-2:]))
        self.assertEqual(row["after"][target]["status"], task["target_status"])

    def test_04_completed_create_prefix_releases_without_model(self):
        task = self.task("post_recovery_immediate_release")
        row = c.run_episode(task, runtime_v1.ScriptedProvider([]), variant="synthetic")
        self.assertTrue(row["prefix_completion_release"])
        self.assertEqual(row["rounds"], [])
        self.assertEqual(row["invalid_outputs"], 0)

    def test_05_missing_visible_search_value_closes_recovery(self):
        task = self.task("exact_search_recovery")
        task["request"] = task["request"].replace(task["trusted_search_filters"]["query"], "[hidden]")
        episode = c.TrustedCompletionEpisode(task)
        result, _ = episode.propose(task["fault_plan"]["call"], "model_proposal")
        self.assertFalse(result["allowed"])
        self.assertFalse(any(event["provenance"] == "guard_recovery" for event in episode.events))

    def test_06_missing_visible_update_target_closes_recovery(self):
        task = self.task("trusted_target_reread_recovery")
        target = task["initial_state"]["tickets"][0]["ticket_id"]
        task["request"] = task["request"].replace(target, "[hidden]")
        episode = c.TrustedCompletionEpisode(task)
        result, _ = episode.propose(task["fault_plan"]["call"], "model_proposal")
        self.assertFalse(result["allowed"])
        self.assertFalse(any(event["provenance"] == "guard_recovery" for event in episode.events))

    def test_07_frozen_v1_6_is_not_modified(self):
        self.assertEqual(population.sha(population.ROOT /
                         "extensions/ticket_runtime_exact_create_recovery_v1_6/core.py"),
                         "35e16bff1f9c763703bfab2cb07ecc694b9073ddf1cd24665879c9ac7733f580")


if __name__ == "__main__":
    unittest.main(verbosity=2)

