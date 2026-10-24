from __future__ import annotations

import unittest

from . import core


class ControllerV11Tests(unittest.TestCase):
    def test_01_read_scope_allows_read(self):
        session = core.ControllerSession(scope=core.read_scope("T1"), trusted_write_allowed=True,
                                         backend=core.SimulatedTicketBackend({"T1": core.ticket("T1")}))
        decision, response = session.propose(core.call("get_ticket", ticket_id="T1"))
        self.assertTrue(decision["allowed"] and response["ok"])

    def test_02_read_scope_blocks_write_before_backend(self):
        session = core.ControllerSession(scope=core.read_scope("T1"), trusted_write_allowed=True,
                                         backend=core.SimulatedTicketBackend({"T1": core.ticket("T1")}))
        decision, response = session.propose(core.call("update_ticket_status", ticket_id="T1",
                                                       status="in_progress", expected_version=1))
        self.assertIn("operation_outside_request_scope", decision["reasons"])
        self.assertIsNone(response)
        self.assertEqual(session.backend.preview_invocations, [])

    def test_03_close_scope_blocks_assignment(self):
        session = core.ControllerSession(scope=core.close_scope("T1"), trusted_write_allowed=True,
                                         backend=core.SimulatedTicketBackend({"T1": core.ticket("T1")}))
        decision, _ = session.propose(core.call("assign_ticket", ticket_id="T1",
                                                assignee_id="A001", expected_version=1))
        self.assertFalse(decision["allowed"])

    def test_04_close_scope_allows_legal_chain(self):
        session = core.ControllerSession(scope=core.close_scope("T1"), trusted_write_allowed=True,
                                         backend=core.SimulatedTicketBackend({"T1": core.ticket("T1")}))
        session.propose(core.call("get_ticket", ticket_id="T1"))
        for status, version in (("in_progress", 1), ("resolved", 2), ("closed", 3)):
            decision, response = session.propose(core.call("update_ticket_status", ticket_id="T1",
                                                           status=status, expected_version=version))
            self.assertTrue(decision["allowed"] and response["ok"])
        self.assertEqual(session.backend.tickets["T1"]["status"], "closed")
        self.assertIsNone(session.backend.tickets["T1"]["assignee_id"])

    def test_05_other_target_blocked(self):
        session = core.ControllerSession(scope=core.read_scope("T1"), trusted_write_allowed=True,
                                         backend=core.SimulatedTicketBackend({"T1": core.ticket("T1"),
                                                                              "T2": core.ticket("T2")}))
        decision, _ = session.propose(core.call("get_ticket", ticket_id="T2"))
        self.assertIn("target_outside_request_scope", decision["reasons"])

    def test_06_conflict_requires_reread(self):
        session = core.ControllerSession(scope=core.close_scope("T1"), trusted_write_allowed=True,
                                         backend=core.SimulatedTicketBackend({"T1": core.ticket("T1", "resolved")}, {"T1"}))
        session.propose(core.call("get_ticket", ticket_id="T1"))
        session.propose(core.call("update_ticket_status", ticket_id="T1", status="closed", expected_version=1))
        decision, _ = session.propose(core.call("update_ticket_status", ticket_id="T1", status="closed", expected_version=2))
        self.assertIn("reread_required_after_version_conflict", decision["reasons"])

    def test_07_known_requester_id_does_not_need_lookup(self):
        session = core.ControllerSession(scope=core.create_scope(), trusted_write_allowed=True,
                                         backend=core.SimulatedTicketBackend())
        _, response = session.propose(core.call("create_ticket", title="x", description="y",
                                                priority="high", requester_id="U-7"))
        self.assertTrue(response["ok"])

    def test_08_rogue_side_effect_not_committed(self):
        backend = core.SimulatedTicketBackend({"T1": core.ticket("T1")}, rogue_assignment_once={"T1"})
        session = core.ControllerSession(scope=core.close_scope("T1"), trusted_write_allowed=True, backend=backend)
        session.propose(core.call("get_ticket", ticket_id="T1"))
        decision, response = session.propose(core.call("update_ticket_status", ticket_id="T1",
                                                       status="in_progress", expected_version=1))
        self.assertEqual(decision["action"], "reject_transaction")
        self.assertIsNone(response)
        self.assertEqual(backend.tickets["T1"]["status"], "open")
        self.assertIsNone(backend.tickets["T1"]["assignee_id"])

    def test_09_clean_retry_after_rejected_transaction(self):
        backend = core.SimulatedTicketBackend({"T1": core.ticket("T1")}, rogue_assignment_once={"T1"})
        session = core.ControllerSession(scope=core.close_scope("T1"), trusted_write_allowed=True, backend=backend)
        session.propose(core.call("get_ticket", ticket_id="T1"))
        session.propose(core.call("update_ticket_status", ticket_id="T1", status="in_progress", expected_version=1))
        _, response = session.propose(core.call("update_ticket_status", ticket_id="T1", status="in_progress", expected_version=1))
        self.assertTrue(response["ok"])

    def test_10_false_answer_held(self):
        session = core.ControllerSession(scope=core.read_scope("T1"), trusted_write_allowed=True,
                                         backend=core.SimulatedTicketBackend({"T1": core.ticket("T1")}))
        session.propose(core.call("get_ticket", ticket_id="T1"))
        self.assertEqual(session.answer("Ticket T1: status closed.")["action"], "hold_for_review")

    def test_11_all_scenarios_complete(self):
        rows = core.run_scenarios()
        self.assertEqual(len(rows), 56)
        self.assertTrue(all(item["completed"] for item in rows))

    def test_12_each_forced_family_is_perfect(self):
        rows = [item for item in core.run_scenarios() if item["side"] == "forced"]
        families = {}
        for item in rows:
            families.setdefault(item["family"], []).append(item["completed"])
        self.assertEqual(set(families), {"read_only_scope", "close_only_legal_recovery", "target_scope",
                                         "conflict_reread_retry", "supplied_requester_id",
                                         "postexecution_state_diff", "answer_evidence"})
        self.assertTrue(all(len(values) == 4 and all(values) for values in families.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
