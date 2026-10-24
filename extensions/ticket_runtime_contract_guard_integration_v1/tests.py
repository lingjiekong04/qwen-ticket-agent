"""CPU-only adapter tests; no model, CUDA or live tool endpoint."""

from __future__ import annotations

import copy
import json
import unittest

from extensions.ticket_runtime_contract_guard_v1_1 import core as guard
from . import core


def ticket_task(*, profile="json", target=None, fault=None):
    tid = "INT-101"
    tools = ["get_ticket"] if target is None else ["get_ticket", "update_ticket_status"]
    return {"id": "integration-task", "family": "synthetic", "entry_kind": "direct_control",
            "locale": "zh", "profile": profile, "request": "处理并汇报 INT-101。",
            "trusted_write_allowed": target is not None,
            "request_scope": {"allowed_tools": tools, "target_ticket_ids": [tid],
                              "allowed_mutations": ({} if target is None else
                                                    {"update_ticket_status": ["status"]})},
            "answer_contract": {"required_answer_fields": ["ticket_id", "status", "version",
                                                                    "priority", "assignee_id"],
                                "output_profile": profile},
            "initial_state": {"tickets": [{"ticket_id": tid, "title": "Issue",
                                               "description": "Details", "priority": "high",
                                               "requester_id": "U-1", "assignee_id": None,
                                               "status": "open", "version": 1}],
                              "users": [{"user_id": "U-1", "email": "u@example.test", "name": "User"}],
                              "agents": [{"agent_id": "A-1", "name": "Agent"}], "articles": []},
            "fault_plan": fault, "target_status": target}


class RuntimeIntegrationTests(unittest.TestCase):
    def test_01_workflow_binding_rejects_scope_mismatch(self):
        task = ticket_task()
        task["answer_contract"]["required_entity_ids"] = ["INT-101"]
        task["request_scope"]["target_ticket_ids"] = ["OTHER"]
        with self.assertRaises(ValueError):
            core.RuntimeEpisode(task)

    def test_02_only_allowed_tool_schemas_are_exposed(self):
        value = json.loads(core.payload([], ticket_task())["tools"])
        self.assertEqual([item["function"]["name"] for item in value], ["get_ticket"])

    def test_03_invalid_output_is_feedback_not_terminal(self):
        episode = core.RuntimeEpisode(ticket_task())
        result = episode.process('<tool_call>{"name":"get_ticket"}</tool_call>')
        self.assertEqual(result["kind"], "feedback")
        self.assertEqual(episode.invalid_outputs, 1)
        self.assertIsNone(episode.released_text)

    def test_04_model_draft_is_replaced_by_verified_json(self):
        episode = core.RuntimeEpisode(ticket_task())
        episode.process(guard.wrapped("get_ticket", ticket_id="INT-101"))
        self.assertEqual(json.loads(episode.released_text)["status"], "open")
        self.assertNotIn("closed", episode.released_text)

    def test_05_sentence_renderer_is_complete(self):
        episode = core.RuntimeEpisode(ticket_task(profile="sentence"))
        episode.process(guard.wrapped("get_ticket", ticket_id="INT-101"))
        self.assertIn("工单ID=INT-101", episode.released_text)
        self.assertIn("负责人=未分配", episode.released_text)

    def test_06_explicit_action_renderer_separates_action_and_result(self):
        episode = core.RuntimeEpisode(ticket_task(profile="explicit_actions", target="in_progress"))
        episode.process(guard.wrapped("get_ticket", ticket_id="INT-101"))
        episode.process(guard.wrapped("update_ticket_status", ticket_id="INT-101",
                                      status="in_progress", expected_version=1))
        self.assertIn("已执行写入 1 次；核实结果：", episode.released_text)

    def test_07_premature_final_is_held_then_recovers(self):
        provider = core.ScriptedProvider([
            "已经关闭。", guard.wrapped("get_ticket", ticket_id="INT-101")])
        record = core.run_episode(ticket_task(), provider, variant="synthetic")
        self.assertEqual(record["termination"], "final_response")
        self.assertEqual(len(record["rounds"]), 2)
        self.assertEqual(json.loads(record["final_text"])["status"], "open")

    def test_08_invalid_output_then_valid_call_recovers(self):
        provider = core.ScriptedProvider([
            '<tool_call>{"name":"get_ticket"}</tool_call>',
            guard.wrapped("get_ticket", ticket_id="INT-101")])
        record = core.run_episode(ticket_task(), provider, variant="synthetic")
        self.assertEqual(record["termination"], "final_response")
        self.assertEqual(record["invalid_outputs"], 1)

    def test_09_illegal_status_jump_is_blocked_then_corrected(self):
        provider = core.ScriptedProvider([
            guard.wrapped("get_ticket", ticket_id="INT-101"),
            guard.wrapped("update_ticket_status", ticket_id="INT-101", status="closed", expected_version=1),
            guard.wrapped("update_ticket_status", ticket_id="INT-101", status="in_progress", expected_version=1)])
        record = core.run_episode(ticket_task(target="in_progress"), provider, variant="synthetic")
        self.assertEqual(record["termination"], "final_response")
        self.assertEqual(record["after"]["INT-101"]["status"], "in_progress")
        self.assertFalse(record["events"][1]["reached_backend"])

    def test_10_no_model_final_round_is_needed_after_goal(self):
        provider = core.ScriptedProvider([guard.wrapped("get_ticket", ticket_id="INT-101")])
        record = core.run_episode(ticket_task(), provider, variant="synthetic")
        self.assertEqual(record["termination"], "final_response")
        self.assertEqual(len(record["rounds"]), 1)

    def test_11_live_task_inputs_are_deep_copied(self):
        task = ticket_task()
        original = copy.deepcopy(task)
        core.RuntimeEpisode(task)
        self.assertEqual(task, original)

    def test_12_all_eight_tool_schemas_are_available_to_adapter(self):
        self.assertEqual({item["function"]["name"] for item in core.tool_schemas(sorted(guard.TOOL_ARGS))},
                         set(guard.TOOL_ARGS))


if __name__ == "__main__":
    unittest.main()
