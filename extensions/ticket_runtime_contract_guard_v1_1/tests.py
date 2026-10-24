"""CPU-only unit and closed-loop tests using synthetic in-memory data."""

from __future__ import annotations

import copy
import json
import unittest

from extensions.ticket_runtime_controller_v1_1 import core as frozen_controller
from .core import (GuardedSession, TrustedWorkflow, call, run_closed_loop,
                   sample_backend, validate_call, wrapped)


class ContractGuardV11Tests(unittest.TestCase):
    def session(self, contract: TrustedWorkflow, **backend_kwargs):
        return GuardedSession(contract=contract, backend=sample_backend(**backend_kwargs),
                              trusted_write_allowed=True)

    def test_01_all_eight_tool_schemas_validate(self):
        good = [
            call("lookup_user", email="a@b.test"), call("search_tickets"),
            call("get_ticket", ticket_id="T1"),
            call("create_ticket", title="x", description="y", requester_id="U1", priority="high"),
            call("assign_ticket", ticket_id="T1", assignee_id="A1", expected_version=1),
            call("update_ticket_status", ticket_id="T1", status="in_progress", expected_version=1),
            call("search_kb", query="vpn"), call("get_kb_article", article_id="K1"),
        ]
        self.assertTrue(all(validate_call(item) is None for item in good))

    def test_02_extra_or_missing_arguments_fail_closed(self):
        self.assertIsNotNone(validate_call(call("lookup_user", email="a", extra="x")))
        self.assertIsNotNone(validate_call(call("assign_ticket", ticket_id="T1", assignee_id="A1")))

    def test_03_lookup_contract_blocks_email_substitution(self):
        session = self.session(TrustedWorkflow("lookup_user", email="mei@example.test"))
        result = session.propose(call("lookup_user", email="alex@example.test"))
        self.assertFalse(result["allowed"])
        self.assertEqual(session.backend.preview_invocations, [])

    def test_04_ticket_search_requires_exact_trusted_filters(self):
        session = self.session(TrustedWorkflow("search_tickets", search_filters={"status": "open"}))
        self.assertFalse(session.propose(call("search_tickets", status="closed"))["allowed"])
        self.assertEqual(session.backend.preview_invocations, [])
        self.assertTrue(session.propose(call("search_tickets", status="open"))["allowed"])

    def test_05_kb_article_must_come_from_search(self):
        session = self.session(TrustedWorkflow("knowledge", kb_query="VPN"))
        self.assertFalse(session.propose(call("get_kb_article", article_id="KB-1"))["allowed"])
        self.assertEqual(session.backend.preview_invocations, [])
        self.assertTrue(session.propose(call("search_kb", query="VPN"))["allowed"])
        self.assertTrue(session.propose(call("get_kb_article", article_id="KB-1"))["allowed"])

    def test_06_create_binds_lookup_identity_and_exact_payload(self):
        contract = TrustedWorkflow("create_ticket", email="mei@example.test",
                                   create_payload={"title": "Printer", "description": "Paper jam", "priority": "normal"})
        session = self.session(contract)
        self.assertEqual(session.propose(call("create_ticket", title="Printer", description="Paper jam",
                                              priority="normal", requester_id="U-7"))["reason"],
                         "lookup_before_create")
        session.propose(call("lookup_user", email="mei@example.test"))
        self.assertFalse(session.propose(call("create_ticket", title="Printer", description="Paper jam",
                                              priority="normal", requester_id="U-8"))["allowed"])
        self.assertTrue(session.propose(call("create_ticket", title="Printer", description="Paper jam",
                                             priority="normal", requester_id="U-7"))["allowed"])
        self.assertTrue(session.render()["operation_completed"])

    def test_07_create_is_exactly_once(self):
        contract = TrustedWorkflow("create_ticket", requester_id="U-7",
                                   create_payload={"title": "Printer", "description": "Paper jam", "priority": "normal"})
        session = self.session(contract)
        proposed = call("create_ticket", title="Printer", description="Paper jam", priority="normal", requester_id="U-7")
        self.assertTrue(session.propose(proposed)["allowed"])
        self.assertEqual(session.propose(proposed)["reason"], "goal_already_satisfied")
        self.assertEqual(len(session.backend.tickets), 3)

    def test_08_assignment_requires_read_current_version_and_exact_agent(self):
        session = self.session(TrustedWorkflow("assign_ticket", target_ticket_id="SYN-101", target_assignee_id="A-1"))
        self.assertEqual(session.propose(call("assign_ticket", ticket_id="SYN-101", assignee_id="A-1",
                                              expected_version=1))["reason"], "read_before_assignment")
        session.propose(call("get_ticket", ticket_id="SYN-101"))
        self.assertFalse(session.propose(call("assign_ticket", ticket_id="SYN-101", assignee_id="A-X",
                                              expected_version=1))["allowed"])
        self.assertTrue(session.propose(call("assign_ticket", ticket_id="SYN-101", assignee_id="A-1",
                                             expected_version=1))["allowed"])

    def test_09_assignment_conflict_latches_until_reread(self):
        session = self.session(TrustedWorkflow("assign_ticket", target_ticket_id="SYN-101", target_assignee_id="A-1"),
                               conflict_once={"SYN-101"})
        session.propose(call("get_ticket", ticket_id="SYN-101"))
        self.assertFalse(session.propose(call("assign_ticket", ticket_id="SYN-101", assignee_id="A-1",
                                              expected_version=1))["allowed"])
        self.assertFalse(session.propose(call("assign_ticket", ticket_id="SYN-101", assignee_id="A-1",
                                              expected_version=1))["allowed"])
        session.propose(call("get_ticket", ticket_id="SYN-101"))
        self.assertTrue(session.propose(call("assign_ticket", ticket_id="SYN-101", assignee_id="A-1",
                                             expected_version=2))["allowed"])

    def test_10_status_chain_stops_at_goal(self):
        session = self.session(TrustedWorkflow("update_status", target_ticket_id="SYN-101", target_status="resolved"))
        session.propose(call("get_ticket", ticket_id="SYN-101"))
        session.propose(call("update_ticket_status", ticket_id="SYN-101", status="in_progress", expected_version=1))
        session.propose(call("update_ticket_status", ticket_id="SYN-101", status="resolved", expected_version=2))
        result = session.propose(call("update_ticket_status", ticket_id="SYN-101", status="closed", expected_version=3))
        self.assertEqual(result["reason"], "goal_already_satisfied")
        self.assertEqual(session.backend.tickets["SYN-101"]["status"], "resolved")

    def test_11_transactional_rogue_diff_is_not_committed(self):
        session = self.session(TrustedWorkflow("update_status", target_ticket_id="SYN-101", target_status="in_progress"),
                               rogue_assignment_once={"SYN-101"})
        session.propose(call("get_ticket", ticket_id="SYN-101"))
        result = session.propose(call("update_ticket_status", ticket_id="SYN-101", status="in_progress", expected_version=1))
        self.assertFalse(result["allowed"])
        self.assertEqual(session.backend.tickets["SYN-101"]["status"], "open")
        self.assertIsNone(session.backend.tickets["SYN-101"]["assignee_id"])

    def test_12_model_final_is_discarded_for_verified_renderer(self):
        session = self.session(TrustedWorkflow("lookup_user", email="mei@example.test"))
        session.propose(call("lookup_user", email="mei@example.test"))
        result = session.dispatch("user_id=FAKE")
        self.assertTrue(result["model_draft_discarded"])
        self.assertEqual(json.loads(result["text"])["user_id"], "U-7")

    def test_13_read_operations_do_not_change_ticket_state(self):
        for contract, proposed in [
            (TrustedWorkflow("lookup_user", email="mei@example.test"), call("lookup_user", email="mei@example.test")),
            (TrustedWorkflow("search_tickets", search_filters={}), call("search_tickets")),
            (TrustedWorkflow("knowledge", kb_query="VPN"), call("search_kb", query="VPN")),
        ]:
            with self.subTest(operation=contract.operation):
                session = self.session(contract)
                before = copy.deepcopy(session.backend.tickets)
                self.assertTrue(session.propose(proposed)["allowed"])
                self.assertEqual(before, session.backend.tickets)

    def test_14_parser_dispatches_multiple_provenance_bound_calls(self):
        contract = TrustedWorkflow("create_ticket", email="mei@example.test",
                                   create_payload={"title": "Printer", "description": "Paper jam", "priority": "normal"})
        session = self.session(contract)
        output = wrapped("lookup_user", email="mei@example.test") + wrapped(
            "create_ticket", title="Printer", description="Paper jam", priority="normal", requester_id="U-7")
        self.assertEqual([item["allowed"] for item in session.dispatch(output)["results"]], [True, True])

    def test_15_closed_loop_population_complete(self):
        rows = run_closed_loop()
        self.assertEqual(len(rows), 17)
        self.assertEqual(len({item["id"] for item in rows}), 17)
        self.assertTrue(all(item["scenario_passed"] for item in rows))
        self.assertEqual(sum(item["completed"] for item in rows), 16)
        self.assertTrue(all(item["preexecution_blocks_reached_backend"] == 0 for item in rows))
        self.assertTrue(all(item["unrequested_state_changes_committed"] == 0 for item in rows))

    def test_16_frozen_controller_hash_is_unchanged(self):
        self.assertEqual(frozen_controller.sha(frozen_controller.Path(frozen_controller.__file__)),
                         "be3622b92553bc15550949f7e035e5e9a5679e1bf3c430fa153735cc1b6ee3f2")


if __name__ == "__main__":
    unittest.main()
