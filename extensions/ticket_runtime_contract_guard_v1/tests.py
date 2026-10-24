"""Synthetic CPU tests. No development or consumed-holdout tasks are loaded."""

from __future__ import annotations

import json
import unittest

from extensions.ticket_runtime_controller_v1_1 import core as controller
from .core import GuardedSession, TrustedContract, parse_model_output


def wrapped(name: str, **args) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"


def make_session(*, goal=None, profile="json", backend=None, write=True, fields=None):
    tid = "SYN-901"
    scope = controller.close_scope(tid) if goal else controller.read_scope(tid)
    contract = TrustedContract(target_ticket_id=tid,
                               required_fields=tuple(fields or ("ticket_id", "status", "version",
                                                                "priority", "assignee_id")),
                               profile=profile, target_status=goal, locale="zh")
    backend = backend or controller.SimulatedTicketBackend({tid: controller.ticket(tid)})
    return GuardedSession(contract=contract, scope=scope, backend=backend,
                          trusted_write_allowed=write)


class ContractGuardTests(unittest.TestCase):
    def test_01_strict_output_parser(self):
        self.assertEqual(parse_model_output(wrapped("get_ticket", ticket_id="SYN-901"))[0], "calls")
        for bad in ('<tool_call>{"name":"get_ticket","name":"get_ticket","arguments":{}}</tool_call>',
                    '<tool_call>{"name":"get_ticket","arguments":{"ticket_id":NaN}}</tool_call>',
                    '<tool_call>{}</tool_call> trailing text', '<tool_call>{}'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_model_output(bad)

    def test_02_invalid_schema_never_reaches_backend(self):
        session = make_session()
        result = session.dispatch(wrapped("get_ticket", ticket_id="SYN-901", extra="x"))
        self.assertFalse(result["results"][0]["allowed"])
        self.assertEqual(session.controller.backend.preview_invocations, [])

    def test_03_read_only_write_is_blocked(self):
        session = make_session()
        result = session.propose(controller.call("update_ticket_status", ticket_id="SYN-901",
                                                 status="in_progress", expected_version=1))
        self.assertFalse(result["allowed"])
        self.assertEqual(session.controller.backend.preview_invocations, [])

    def test_04_final_draft_cannot_fabricate_or_break_json(self):
        session = make_session()
        session.propose(controller.call("get_ticket", ticket_id="SYN-901"))
        result = session.dispatch("Ticket SYN-901 is closed and assigned to A999.")
        self.assertTrue(result["allowed"])
        self.assertTrue(result["model_draft_discarded"])
        self.assertEqual(json.loads(result["text"]), {
            "ticket_id": "SYN-901", "status": "open", "version": 1,
            "priority": "high", "assignee_id": None})

    def test_05_no_observation_means_no_state_claim(self):
        result = make_session().dispatch("Everything is done.")
        self.assertFalse(result["allowed"])
        self.assertEqual(result["reason"], "no_current_verified_state")

    def test_06_goal_stops_overshoot_in_same_model_message(self):
        session = make_session(goal="resolved")
        session.propose(controller.call("get_ticket", ticket_id="SYN-901"))
        text = "".join((wrapped("update_ticket_status", ticket_id="SYN-901",
                                status="in_progress", expected_version=1),
                        wrapped("update_ticket_status", ticket_id="SYN-901",
                                status="resolved", expected_version=2),
                        wrapped("update_ticket_status", ticket_id="SYN-901",
                                status="closed", expected_version=3)))
        results = session.dispatch(text)["results"]
        self.assertEqual([item["allowed"] for item in results], [True, True, False])
        self.assertEqual(session.controller.backend.tickets["SYN-901"]["status"], "resolved")
        self.assertEqual(len(session.controller.backend.preview_invocations), 3)  # read + two writes
        final = session.render()
        self.assertTrue(final["operation_completed"])
        self.assertEqual(json.loads(final["text"])["status"], "resolved")

    def test_07_write_requires_read_and_next_step(self):
        session = make_session(goal="resolved")
        self.assertEqual(session.propose(controller.call("update_ticket_status", ticket_id="SYN-901",
                                                         status="in_progress", expected_version=1))["reason"],
                         "read_before_status_write")
        session.propose(controller.call("get_ticket", ticket_id="SYN-901"))
        self.assertEqual(session.propose(controller.call("update_ticket_status", ticket_id="SYN-901",
                                                         status="resolved", expected_version=1))["reason"],
                         "write_not_next_step_to_goal")
        self.assertEqual(len(session.controller.backend.preview_invocations), 1)

    def test_08_repeated_no_progress_rejection_halts(self):
        session = make_session(goal="in_progress")
        session.propose(controller.call("get_ticket", ticket_id="SYN-901"))
        bad = controller.call("update_ticket_status", ticket_id="SYN-901",
                              status="closed", expected_version=1)
        self.assertFalse(session.propose(bad)["halted"])
        self.assertTrue(session.propose(bad)["halted"])
        self.assertFalse(session.propose(controller.call("get_ticket", ticket_id="SYN-901"))["allowed"])
        self.assertEqual(len(session.controller.backend.preview_invocations), 1)

    def test_09_conflict_requires_reread(self):
        tid = "SYN-901"
        backend = controller.SimulatedTicketBackend({tid: controller.ticket(tid)}, conflict_once={tid})
        session = make_session(goal="in_progress", backend=backend)
        session.propose(controller.call("get_ticket", ticket_id=tid))
        first = session.propose(controller.call("update_ticket_status", ticket_id=tid,
                                                status="in_progress", expected_version=1))
        self.assertFalse(first["allowed"])
        self.assertTrue(first["backend_executed"])
        retry = session.propose(controller.call("update_ticket_status", ticket_id=tid,
                                                status="in_progress", expected_version=1))
        self.assertFalse(retry["allowed"])
        self.assertFalse(retry["backend_executed"])
        session.propose(controller.call("get_ticket", ticket_id=tid))
        self.assertTrue(session.propose(controller.call("update_ticket_status", ticket_id=tid,
                                                        status="in_progress", expected_version=2))["allowed"])

    def test_10_rogue_diff_rolls_back_and_answer_is_truthful(self):
        tid = "SYN-901"
        backend = controller.SimulatedTicketBackend({tid: controller.ticket(tid)},
                                                     rogue_assignment_once={tid})
        session = make_session(goal="in_progress", backend=backend)
        session.propose(controller.call("get_ticket", ticket_id=tid))
        decision = session.propose(controller.call("update_ticket_status", ticket_id=tid,
                                                   status="in_progress", expected_version=1))
        self.assertFalse(decision["allowed"])
        self.assertTrue(decision["backend_executed"])  # preview only; no commit
        self.assertEqual(backend.tickets[tid]["status"], "open")
        self.assertIsNone(backend.tickets[tid]["assignee_id"])
        answer = session.render()
        self.assertFalse(answer["operation_completed"])
        self.assertEqual(json.loads(answer["text"])["status"], "open")
        self.assertNotIn("operation_completed", json.loads(answer["text"]))

    def test_11_canonical_null_in_sentence(self):
        session = make_session(profile="sentence", fields=("ticket_id", "assignee_id", "status"))
        session.propose(controller.call("get_ticket", ticket_id="SYN-901"))
        text = session.render()["text"]
        self.assertIn("负责人规范ID=未分配", text)
        self.assertIn("状态=open", text)

    def test_12_unsupported_ticket_operations_fail_closed(self):
        session = make_session()
        result = session.propose(controller.call("assign_ticket", ticket_id="SYN-901",
                                                 assignee_id="A001", expected_version=1))
        self.assertFalse(result["allowed"])
        self.assertEqual(session.controller.backend.preview_invocations, [])

    def test_13_scope_or_permission_cannot_be_model_supplied(self):
        tid = "SYN-901"
        with self.assertRaises(ValueError):
            GuardedSession(contract=TrustedContract(tid, ("ticket_id", "status")),
                           scope=controller.close_scope(tid),
                           backend=controller.SimulatedTicketBackend({tid: controller.ticket(tid)}),
                           trusted_write_allowed=True)
        with self.assertRaises(ValueError):
            GuardedSession(contract=TrustedContract(tid, ("ticket_id", "status")),
                           scope=controller.read_scope(tid),
                           backend=controller.SimulatedTicketBackend({tid: controller.ticket(tid)}),
                           trusted_write_allowed="yes")

    def test_14_explicit_actions_separate_execution_and_evidence(self):
        tid = "SYN-901"
        contract = TrustedContract(tid, ("ticket_id", "status", "version"),
                                   "explicit_actions", "in_progress", "zh")
        session = GuardedSession(contract=contract, scope=controller.close_scope(tid),
                                 backend=controller.SimulatedTicketBackend({tid: controller.ticket(tid)}),
                                 trusted_write_allowed=True)
        session.propose(controller.call("get_ticket", ticket_id=tid))
        session.propose(controller.call("update_ticket_status", ticket_id=tid,
                                        status="in_progress", expected_version=1))
        text = session.render()["text"]
        self.assertIn("已执行状态写入 1 次；核实结果：", text)
        self.assertIn("状态=in_progress", text)


if __name__ == "__main__":
    unittest.main()
