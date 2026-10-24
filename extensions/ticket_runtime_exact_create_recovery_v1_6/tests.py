from __future__ import annotations

import copy
import unittest

from extensions.ticket_runtime_contract_guard_v1_1 import core as guard
from extensions.ticket_runtime_exact_create_recovery_v1_6 import core as c
from extensions.ticket_runtime_visible_contract_dev_v1 import core as source


def create_task() -> dict:
    tasks, _ = source.generate()
    return copy.deepcopy(next(task for task in tasks if task["primary_tool"] == "create_ticket"))


def exact(task: dict) -> dict:
    return guard.call("create_ticket", requester_id=task["trusted_requester_id"],
                      **task["trusted_create_payload"])


class ExactCreateRecoveryTests(unittest.TestCase):
    def assert_recovers(self, field: str, value: str):
        task = create_task()
        bad = exact(task)
        bad["arguments"][field] = value
        episode = c.ExactCreatePayloadRecoveryEpisode(task)
        result, response = episode.propose(bad, "model_proposal")
        self.assertTrue(result["allowed"])
        self.assertEqual(response["controller_recovery_call"], exact(task))
        self.assertEqual(len(episode.events), 2)
        self.assertFalse(episode.events[0]["reached_backend"])
        self.assertTrue(episode.events[1]["reached_backend"])

    def test_01_title_mismatch_recovers_exact_payload(self):
        self.assert_recovers("title", "truncated title")

    def test_02_description_mismatch_recovers_exact_payload(self):
        self.assert_recovers("description", "wrong description")

    def test_03_priority_mismatch_recovers_exact_payload(self):
        self.assert_recovers("priority", "high")

    def test_04_requester_mismatch_recovers_exact_payload(self):
        self.assert_recovers("requester_id", "VCP21U-CREATE--OTHER-0")

    def test_05_missing_required_value_is_not_recovered(self):
        task = create_task()
        bad = exact(task)
        del bad["arguments"]["title"]
        episode = c.ExactCreatePayloadRecoveryEpisode(task)
        result, response = episode.propose(bad, "model_proposal")
        self.assertFalse(result["allowed"])
        self.assertNotIn("controller_recovery_call", response)
        self.assertFalse(any(event["reached_backend"] for event in episode.events))

    def test_06_invalid_type_is_not_recovered(self):
        task = create_task()
        bad = exact(task)
        bad["arguments"]["title"] = 7
        episode = c.ExactCreatePayloadRecoveryEpisode(task)
        result, response = episode.propose(bad, "model_proposal")
        self.assertFalse(result["allowed"])
        self.assertNotIn("controller_recovery_call", response)

    def test_07_hidden_trusted_value_closes_recovery(self):
        task = create_task()
        task["request"] = task["request"].replace(task["trusted_create_payload"]["title"], "[hidden]")
        bad = exact(task)
        bad["arguments"]["title"] = "wrong"
        episode = c.ExactCreatePayloadRecoveryEpisode(task)
        result, response = episode.propose(bad, "model_proposal")
        self.assertFalse(result["allowed"])
        self.assertNotIn("controller_recovery_call", response)

    def test_08_exact_normal_call_is_not_relabelled_recovery(self):
        task = create_task()
        episode = c.ExactCreatePayloadRecoveryEpisode(task)
        result, response = episode.propose(exact(task), "model_proposal")
        self.assertTrue(result["allowed"])
        self.assertNotIn("controller_recovery_call", response)
        self.assertEqual(episode.events[0]["provenance"], "model_proposal")

    def test_09_recovery_changes_only_one_new_ticket(self):
        task = create_task()
        bad = exact(task)
        bad["arguments"]["title"] = "wrong"
        episode = c.ExactCreatePayloadRecoveryEpisode(task)
        before = copy.deepcopy(episode.backend.tickets)
        episode.propose(bad, "model_proposal")
        self.assertEqual(len(episode.backend.tickets), len(before) + 1)
        self.assertTrue(all(episode.backend.tickets[key] == value for key, value in before.items()))


if __name__ == "__main__":
    unittest.main()

