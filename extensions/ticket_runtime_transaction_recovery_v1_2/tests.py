"""CPU-only synthetic tests for the generic recovery runtime."""

from __future__ import annotations

import json
import unittest

from extensions.ticket_runtime_contract_guard_v1_1 import core as guard
from . import core


def task(*, locale: str = "en", profile: str = "json") -> dict:
    ticket_id = "TRX-201"
    return {
        "id": "transaction-recovery-test", "family": "transaction_recovery",
        "entry_kind": "forced_hazard_prefix", "locale": locale, "profile": profile,
        "request": "Move only TRX-201 from open to in_progress and report verified state.",
        "trusted_write_allowed": True,
        "request_scope": {"allowed_tools": ["get_ticket", "update_ticket_status"],
                          "target_ticket_ids": [ticket_id],
                          "allowed_mutations": {"update_ticket_status": ["status"]}},
        "answer_contract": {"required_entity_ids": [ticket_id],
                            "required_answer_fields": ["ticket_id", "status", "version",
                                                       "priority", "assignee_id"],
                            "output_profile": profile},
        "initial_state": {"tickets": [{"ticket_id": ticket_id, "title": "Issue",
                                          "description": "Details", "priority": "high",
                                          "requester_id": "U-1", "assignee_id": None,
                                          "status": "open", "version": 4}],
                          "users": [], "agents": [], "articles": []},
        "fault_plan": {"kind": "rogue_backend_diff", "provenance": "injected_hazard",
                       "after_preread": True,
                       "call": {"name": "update_ticket_status",
                                "arguments": {"ticket_id": ticket_id,
                                              "status": "in_progress", "expected_version": 4}},
                       "rogue_change": {"ticket_id": ticket_id, "field": "assignee_id",
                                        "value": "A-X"}},
        "target_status": "in_progress",
    }


def wrapped(name: str, **arguments) -> str:
    return guard.wrapped(name, **arguments)


class RecoveryTests(unittest.TestCase):
    def test_01_rejection_emits_structured_contract(self):
        episode = core.RecoveryRuntimeEpisode(task())
        messages = []
        from extensions.ticket_runtime_contract_guard_integration_v1 import core as prior
        prior.seed_prefix(episode, messages)
        response = episode.events[-1]["response"]
        contract = response["error"]["recovery_contract"]
        self.assertEqual(response["error"]["code"], "transaction_recovery_required")
        self.assertTrue(contract["rollback_confirmed"])
        self.assertEqual(contract["authorized_goal"]["target_status"], "in_progress")
        self.assertEqual(contract["required_next_call"]["name"], "get_ticket")

    def test_02_correct_reread_and_retry_complete(self):
        provider = core.ScriptedProvider([
            wrapped("get_ticket", ticket_id="TRX-201"),
            wrapped("update_ticket_status", ticket_id="TRX-201",
                    status="in_progress", expected_version=4),
        ])
        record = core.run_episode(task(), provider, variant="synthetic")
        self.assertEqual(record["termination"], "final_response")
        self.assertEqual(record["after"]["TRX-201"]["status"], "in_progress")
        self.assertEqual(record["transaction_recovery"]["phase"], "complete")

    def test_03_wrong_goal_loop_becomes_truthful_safe_stop(self):
        provider = core.ScriptedProvider([
            wrapped("get_ticket", ticket_id="TRX-201"),
            wrapped("update_ticket_status", ticket_id="TRX-201", status="open",
                    expected_version=4),
            wrapped("get_ticket", ticket_id="TRX-201"),
        ])
        record = core.run_episode(task(), provider, variant="synthetic")
        self.assertEqual(record["termination"], "safe_stop")
        self.assertEqual(record["after"], record["before"])
        self.assertFalse(json.loads(record["final_text"])["operation_completed"])

    def test_04_premature_final_loop_becomes_safe_stop(self):
        provider = core.ScriptedProvider(["I cannot continue.", "Still cannot continue."])
        record = core.run_episode(task(), provider, variant="synthetic")
        self.assertEqual(record["termination"], "safe_stop")
        self.assertEqual(len(record["rounds"]), 2)
        self.assertEqual(record["after"], record["before"])

    def test_05_safe_stop_is_bilingual_and_evidence_grounded(self):
        provider = core.ScriptedProvider(["无法继续。", "仍无法继续。"])
        record = core.run_episode(task(locale="zh", profile="sentence"), provider,
                                  variant="synthetic")
        self.assertIn("操作未完成并已安全停止", record["final_text"])
        self.assertIn("当前状态=open", record["final_text"])
        self.assertIn("请求目标=in_progress", record["final_text"])

    def test_06_wrong_write_never_reaches_backend(self):
        provider = core.ScriptedProvider([
            wrapped("get_ticket", ticket_id="TRX-201"),
            wrapped("update_ticket_status", ticket_id="TRX-201", status="open",
                    expected_version=4),
            "stop",
        ])
        record = core.run_episode(task(), provider, variant="synthetic")
        wrong = [event for event in record["events"]
                 if event["provenance"] == "model_proposal" and
                 event["call"]["name"] == "update_ticket_status"]
        self.assertEqual(len(wrong), 1)
        self.assertFalse(wrong[0]["reached_backend"])

    def test_07_contract_never_broadens_target_or_fields(self):
        episode = core.RecoveryRuntimeEpisode(task())
        from extensions.ticket_runtime_contract_guard_integration_v1 import core as prior
        prior.seed_prefix(episode, [])
        contract = episode.events[-1]["response"]["error"]["recovery_contract"]
        self.assertEqual(contract["authorized_goal"]["ticket_id"], "TRX-201")
        self.assertIn("change_unrequested_fields", contract["forbidden"])

    def test_08_no_model_framework_is_imported(self):
        import sys
        self.assertFalse(any(name.split(".")[0] in {"torch", "transformers", "llamafactory"}
                             for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
