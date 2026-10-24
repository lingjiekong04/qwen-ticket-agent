from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

from extensions.ticket_runtime_contract_guard_v1_1 import core as guard
from extensions.ticket_runtime_contract_guard_integration_v1 import core as runtime_v1
from . import core


ROOT = Path(__file__).resolve().parents[2]
TASKS = json.loads((ROOT / "data/ticket_runtime_full_workflow_dev_v2_1/tasks.json").read_text(
    encoding="utf-8"))


def task(family: str, entry: str = "forced_hazard_prefix") -> dict:
    return copy.deepcopy(next(item for item in TASKS
                              if item["family"] == family and item["entry_kind"] == entry))


class GeneralRecoveryTests(unittest.TestCase):
    def test_01_kb_block_emits_exact_query_contract(self):
        item = task("search_kb")
        episode = core.GeneralBlockedActionRecoveryEpisode(item)
        runtime_v1.seed_prefix(episode, [])
        contract = episode.events[-1]["response"]["error"]["recovery_contract"]
        self.assertEqual(contract["trigger_family"], "wrong_kb_query")
        self.assertEqual(contract["required_next_call"]["arguments"]["query"],
                         item["trusted_kb_query"])

    def test_02_partial_kb_retry_is_replaced_by_exact_query(self):
        item = task("search_kb")
        matching = [article for article in item["initial_state"]["articles"]
                    if item["trusted_kb_query"].casefold() in article["title"].casefold()]
        outputs = [guard.wrapped("search_kb", query="wrong partial")]
        outputs += [guard.wrapped("get_kb_article", article_id=article["article_id"])
                    for article in sorted(matching, key=lambda row: row["article_id"])
                    [:item["max_articles"]]]
        record = core.run_episode(item, core.ScriptedProvider(outputs), variant="synthetic")
        self.assertEqual(record["termination"], "final_response")
        recovery_events = [event for event in record["events"]
                           if event["provenance"] == "guard_recovery"]
        self.assertEqual(recovery_events[0]["call"]["arguments"]["query"],
                         item["trusted_kb_query"])

    def test_03_redundant_assignment_reread_completes_exact_goal(self):
        item = task("assign_ticket")
        ticket = item["initial_state"]["tickets"][0]
        output = guard.wrapped("get_ticket", ticket_id=ticket["ticket_id"])
        record = core.run_episode(item, core.ScriptedProvider([output]), variant="synthetic")
        self.assertEqual(record["termination"], "final_response")
        self.assertEqual(record["after"][ticket["ticket_id"]]["assignee_id"],
                         item["target_assignee_id"])
        self.assertTrue(any(event["provenance"] == "guard_recovery" and
                            event["call"]["name"] == "assign_ticket"
                            for event in record["events"]))

    def test_04_rejected_actions_never_reach_backend(self):
        for family in ("assign_ticket", "search_kb"):
            item = task(family)
            episode = core.GeneralBlockedActionRecoveryEpisode(item)
            runtime_v1.seed_prefix(episode, [])
            blocked = [event for event in episode.events
                       if (event["controller_decision"] or {}).get("action") == "block"]
            self.assertTrue(blocked)
            self.assertFalse(any(event["reached_backend"] for event in blocked))

    def test_05_direct_kb_path_is_unchanged(self):
        item = task("search_kb", "direct_control")
        matching = [article for article in item["initial_state"]["articles"]
                    if item["trusted_kb_query"].casefold() in article["title"].casefold()]
        outputs = [guard.wrapped("search_kb", query=item["trusted_kb_query"])]
        outputs += [guard.wrapped("get_kb_article", article_id=article["article_id"])
                    for article in sorted(matching, key=lambda row: row["article_id"])
                    [:item["max_articles"]]]
        record = core.run_episode(item, core.ScriptedProvider(outputs), variant="synthetic")
        self.assertEqual(record["termination"], "final_response")
        self.assertIsNone(record["recovery"])

    def test_06_frozen_v1_3_unchanged(self):
        path = ROOT / "extensions/ticket_runtime_blocked_action_recovery_v1_3/core.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "70aae82403935920f0d9d21898094499e7c716cd9c8b634e34a21fdcc05126cf")


if __name__ == "__main__":
    unittest.main()

