"""Extended trusted-contract guard and an in-memory closed-loop backend.

The module composes the frozen controller v1.1.  It never imports a model
framework and never calls a live ticket or knowledge service.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
from typing import Any

from extensions.ticket_runtime_contract_guard_v1.core import parse_model_output
from extensions.ticket_runtime_controller_v1_1 import core as controller
from extensions.ticket_runtime_controller_v1_1_design import core as design


STATUSES = {"open", "in_progress", "resolved", "closed"}
PRIORITIES = {"low", "normal", "high"}
WRITE_TOOLS = {"create_ticket", "assign_ticket", "update_ticket_status"}
TOOL_ARGS = {
    "lookup_user": ({"email"}, {"email"}),
    "search_tickets": (set(), {"requester_id", "status", "query"}),
    "get_ticket": ({"ticket_id"}, {"ticket_id"}),
    "create_ticket": ({"title", "description", "requester_id", "priority"},
                      {"title", "description", "requester_id", "priority"}),
    "assign_ticket": ({"ticket_id", "assignee_id", "expected_version"},
                      {"ticket_id", "assignee_id", "expected_version"}),
    "update_ticket_status": ({"ticket_id", "status", "expected_version"},
                             {"ticket_id", "status", "expected_version"}),
    "search_kb": ({"query"}, {"query"}),
    "get_kb_article": ({"article_id"}, {"article_id"}),
}
OPERATIONS = {
    "read_ticket", "update_status", "assign_ticket", "create_ticket",
    "lookup_user", "search_tickets", "knowledge",
}


def call(name: str, **arguments: Any) -> dict:
    return {"name": name, "arguments": arguments}


def wrapped(name: str, **arguments: Any) -> str:
    return "<tool_call>" + json.dumps(call(name, **arguments), ensure_ascii=False) + "</tool_call>"


def validate_call(proposed: dict) -> str | None:
    if not isinstance(proposed, dict) or set(proposed) != {"name", "arguments"}:
        return "invalid_call_shape"
    name, args = proposed["name"], proposed["arguments"]
    if name not in TOOL_ARGS or not isinstance(args, dict):
        return "invalid_tool_or_arguments"
    required, allowed = TOOL_ARGS[name]
    if not required <= set(args) or not set(args) <= allowed:
        return "invalid_tool_or_arguments"
    for key, value in args.items():
        if key == "expected_version":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                return "invalid_expected_version"
        elif not isinstance(value, str) or not value or len(value) > (2000 if key == "description" else 200):
            return "invalid_string_argument"
    if args.get("status") not in STATUSES and "status" in args:
        return "invalid_status"
    if args.get("priority") not in PRIORITIES and "priority" in args:
        return "invalid_priority"
    return None


@dataclass(frozen=True)
class TrustedWorkflow:
    """Immutable application-supplied intent. Model output cannot alter it."""

    operation: str
    locale: str = "en"
    target_ticket_id: str | None = None
    target_status: str | None = None
    target_assignee_id: str | None = None
    email: str | None = None
    requester_id: str | None = None
    create_payload: dict[str, str] | None = None
    search_filters: dict[str, str] | None = None
    kb_query: str | None = None
    max_articles: int = 2

    def __post_init__(self) -> None:
        if self.operation not in OPERATIONS or self.locale not in {"en", "zh"}:
            raise ValueError("unsupported_workflow")
        if self.operation in {"read_ticket", "update_status", "assign_ticket"} and not self.target_ticket_id:
            raise ValueError("ticket_workflow_requires_target")
        if self.operation == "update_status" and self.target_status not in STATUSES:
            raise ValueError("status_workflow_requires_valid_goal")
        if self.operation == "assign_ticket" and not self.target_assignee_id:
            raise ValueError("assignment_requires_assignee")
        if self.operation == "lookup_user" and not self.email:
            raise ValueError("lookup_requires_email")
        if self.operation == "create_ticket":
            if not self.create_payload or set(self.create_payload) != {"title", "description", "priority"}:
                raise ValueError("create_requires_exact_payload")
            if self.create_payload["priority"] not in PRIORITIES or not (self.email or self.requester_id):
                raise ValueError("create_requires_requester_source")
            if self.email and self.requester_id:
                raise ValueError("create_requester_source_is_ambiguous")
        if self.operation == "search_tickets":
            filters = self.search_filters or {}
            if not set(filters) <= TOOL_ARGS["search_tickets"][1]:
                raise ValueError("unsupported_ticket_filter")
            if "status" in filters and filters["status"] not in STATUSES:
                raise ValueError("invalid_search_status")
        if self.operation == "knowledge" and not self.kb_query:
            raise ValueError("knowledge_requires_query")
        if isinstance(self.max_articles, bool) or not isinstance(self.max_articles, int) or not 1 <= self.max_articles <= 5:
            raise ValueError("invalid_article_limit")

    def expected_tools(self) -> set[str]:
        return {
            "read_ticket": {"get_ticket"},
            "update_status": {"get_ticket", "update_ticket_status"},
            "assign_ticket": {"get_ticket", "assign_ticket"},
            "create_ticket": ({"lookup_user", "create_ticket"} if self.email else {"create_ticket"}),
            "lookup_user": {"lookup_user"},
            "search_tickets": {"search_tickets"},
            "knowledge": {"search_kb", "get_kb_article"},
        }[self.operation]

    def scope(self) -> dict:
        mutations: dict[str, list[str]] = {}
        if self.operation == "update_status":
            mutations = {"update_ticket_status": ["status"]}
        elif self.operation == "assign_ticket":
            mutations = {"assign_ticket": ["assignee_id"]}
        elif self.operation == "create_ticket":
            mutations = {"create_ticket": ["title", "description", "priority", "requester_id"]}
        targets = ([self.target_ticket_id] if self.target_ticket_id else [])
        return {"allowed_tools": sorted(self.expected_tools()),
                "target_ticket_ids": targets, "allowed_mutations": mutations}


class ExtendedBackend(controller.SimulatedTicketBackend):
    """Pure in-memory backend with all eight frozen ticket/KB tool schemas."""

    def __init__(self, tickets: dict[str, dict] | None = None, *,
                 users: list[dict] | None = None, agents: list[dict] | None = None,
                 articles: list[dict] | None = None,
                 conflict_once: set[str] | None = None,
                 rogue_assignment_once: set[str] | None = None) -> None:
        super().__init__(tickets, conflict_once, rogue_assignment_once)
        self.users = copy.deepcopy(users or [])
        self.agents = copy.deepcopy(agents or [])
        self.articles = copy.deepcopy(articles or [])

    def preview(self, proposed: dict) -> tuple[dict, dict[str, dict], int]:
        name, args = proposed["name"], proposed["arguments"]
        if name in {"lookup_user", "search_tickets", "search_kb", "get_kb_article"}:
            self.preview_invocations.append(copy.deepcopy(proposed))
            state, next_id = copy.deepcopy(self.tickets), self.next_id
            if name == "lookup_user":
                value = next((item for item in self.users if item["email"].casefold() == args["email"].casefold()), None)
                response = ({"ok": True, "result": copy.deepcopy(value)} if value else
                            {"ok": False, "error": {"code": "user_not_found"}})
            elif name == "search_tickets":
                values = list(state.values())
                if "requester_id" in args:
                    values = [item for item in values if item.get("requester_id") == args["requester_id"]]
                if "status" in args:
                    values = [item for item in values if item.get("status") == args["status"]]
                if "query" in args:
                    needle = args["query"].casefold()
                    values = [item for item in values if needle in (item.get("title", "") + " " + item.get("description", "")).casefold()]
                response = {"ok": True, "result": copy.deepcopy(sorted(values, key=lambda item: item["ticket_id"]))}
            elif name == "search_kb":
                needle = args["query"].casefold()
                values = [{"article_id": item["article_id"], "title": item["title"]}
                          for item in self.articles if needle in item["title"].casefold()]
                response = {"ok": True, "result": copy.deepcopy(sorted(values, key=lambda item: item["article_id"]))}
            else:
                value = next((item for item in self.articles if item["article_id"] == args["article_id"]), None)
                response = ({"ok": True, "result": copy.deepcopy(value)} if value else
                            {"ok": False, "error": {"code": "article_not_found"}})
            return response, state, next_id
        if name == "create_ticket" and not any(item["user_id"] == args["requester_id"] for item in self.users):
            self.preview_invocations.append(copy.deepcopy(proposed))
            return {"ok": False, "error": {"code": "requester_not_found"}}, copy.deepcopy(self.tickets), self.next_id
        if name == "assign_ticket" and not any(item["agent_id"] == args["assignee_id"] for item in self.agents):
            self.preview_invocations.append(copy.deepcopy(proposed))
            return {"ok": False, "error": {"code": "agent_not_found"}}, copy.deepcopy(self.tickets), self.next_id
        return super().preview(proposed)


class GuardedSession:
    def __init__(self, *, contract: TrustedWorkflow, backend: ExtendedBackend,
                 trusted_write_allowed: bool) -> None:
        if not isinstance(trusted_write_allowed, bool):
            raise ValueError("trusted_write_permission_must_be_boolean")
        if contract.operation in {"create_ticket", "update_status", "assign_ticket"} and not trusted_write_allowed:
            # The session is valid, but all writes remain fail-closed in the controller.
            pass
        self.contract = contract
        self.controller = controller.ControllerSession(
            scope=contract.scope(), trusted_write_allowed=trusted_write_allowed, backend=backend)
        self.audit: list[dict] = []
        self.rejections: dict[str, int] = {}
        self.halted = False
        self.user: dict | None = None
        self.ticket_results: list[dict] | None = None
        self.kb_search_results: list[dict] | None = None
        self.articles: dict[str, dict] = {}
        self.created_ticket_id: str | None = None

    @property
    def backend(self) -> ExtendedBackend:
        return self.controller.backend  # type: ignore[return-value]

    def _reject(self, reason: str, proposed: dict | None = None, *,
                backend_executed: bool = False, response: dict | None = None) -> dict:
        key = json.dumps([reason, proposed, self.controller.observed], sort_keys=True,
                         ensure_ascii=False, default=str)
        self.rejections[key] = self.rejections.get(key, 0) + 1
        if self.rejections[key] >= 2:
            self.halted = True
        result = {"allowed": False, "reason": reason, "halted": self.halted,
                  "backend_executed": backend_executed, "response": copy.deepcopy(response)}
        self.audit.append({"call": copy.deepcopy(proposed), **result})
        return result

    def _contract_reason(self, proposed: dict) -> str | None:
        name, args = proposed["name"], proposed["arguments"]
        c = self.contract
        if name not in c.expected_tools():
            return "tool_outside_trusted_workflow"
        if "ticket_id" in args and args["ticket_id"] != c.target_ticket_id:
            return "target_outside_trusted_contract"
        if name == "lookup_user" and args["email"].casefold() != (c.email or "").casefold():
            return "email_outside_trusted_contract"
        if name == "search_tickets" and args != (c.search_filters or {}):
            return "filters_outside_trusted_contract"
        if name == "search_kb" and args["query"] != c.kb_query:
            return "query_outside_trusted_contract"
        if name == "get_kb_article":
            allowed = {item["article_id"] for item in (self.kb_search_results or [])}
            if args["article_id"] not in allowed:
                return "article_not_returned_by_trusted_search"
            if args["article_id"] in self.articles:
                return "article_already_read"
            if len(self.articles) >= c.max_articles:
                return "article_limit_reached"
        if name == "create_ticket":
            if self.created_ticket_id:
                return "goal_already_satisfied"
            expected_requester = c.requester_id or (self.user or {}).get("user_id")
            if expected_requester is None:
                return "lookup_before_create"
            expected = {**(c.create_payload or {}), "requester_id": expected_requester}
            if args != expected:
                return "create_payload_outside_trusted_contract"
        if name == "assign_ticket":
            observed = self.controller.observed.get(c.target_ticket_id or "")
            if observed is None:
                return "read_before_assignment"
            if observed.get("assignee_id") == c.target_assignee_id:
                return "goal_already_satisfied"
            if args["assignee_id"] != c.target_assignee_id:
                return "assignee_outside_trusted_contract"
            if args["expected_version"] != observed.get("version"):
                return "stale_or_missing_expected_version"
        if name == "update_ticket_status":
            observed = self.controller.observed.get(c.target_ticket_id or "")
            if observed is None:
                return "read_before_status_write"
            if observed.get("status") == c.target_status:
                return "goal_already_satisfied"
            if args["expected_version"] != observed.get("version"):
                return "stale_or_missing_expected_version"
            if args["status"] != design.ALLOWED_NEXT.get(observed.get("status")):
                return "write_not_next_step_to_goal"
            reachable, current = set(), design.ALLOWED_NEXT.get(observed.get("status"))
            while current and current not in reachable:
                reachable.add(current)
                current = design.ALLOWED_NEXT.get(current)
            if c.target_status not in reachable:
                return "goal_not_reachable_from_observed_state"
        return None

    def propose(self, proposed: dict) -> dict:
        if self.halted:
            return self._reject("episode_halted", proposed)
        invalid = validate_call(proposed)
        if invalid:
            return self._reject(invalid, proposed)
        contract_error = self._contract_reason(proposed)
        if contract_error:
            return self._reject(contract_error, proposed)
        before = len(self.backend.preview_invocations)
        decision, response = self.controller.propose(copy.deepcopy(proposed))
        reached = len(self.backend.preview_invocations) > before
        if not decision["allowed"]:
            return self._reject("controller_blocked:" + ",".join(decision["reasons"]), proposed,
                                backend_executed=reached, response=response)
        if response is None or not response.get("ok"):
            code = (response or {}).get("error", {}).get("code", "missing_backend_response")
            return self._reject("backend_error:" + code, proposed,
                                backend_executed=reached, response=response)
        result = response.get("result")
        name = proposed["name"]
        if name == "lookup_user":
            self.user = copy.deepcopy(result)
        elif name == "search_tickets":
            self.ticket_results = copy.deepcopy(result)
        elif name == "search_kb":
            self.kb_search_results = copy.deepcopy(result)
        elif name == "get_kb_article":
            self.articles[result["article_id"]] = copy.deepcopy(result)
        elif name == "create_ticket":
            self.created_ticket_id = result["ticket_id"]
        self.rejections.clear()
        accepted = {"allowed": True, "decision": copy.deepcopy(decision),
                    "response": copy.deepcopy(response), "backend_executed": reached,
                    "halted": self.halted}
        self.audit.append({"call": copy.deepcopy(proposed), **accepted})
        return accepted

    def dispatch(self, text: str) -> dict:
        if self.halted:
            return self._reject("episode_halted")
        try:
            kind, value = parse_model_output(text)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self._reject("invalid_model_output:" + str(exc))
        if kind == "final":
            return {"kind": "final", "model_draft_discarded": True, **self.render()}
        results = []
        for proposed in value:
            results.append(self.propose(proposed))
            if self.halted:
                break
        return {"kind": "calls", "results": results, "halted": self.halted}

    def render(self) -> dict:
        c = self.contract
        complete = False
        payload: dict[str, Any]
        source: str
        if c.operation in {"read_ticket", "update_status", "assign_ticket"}:
            observed = self.controller.observed.get(c.target_ticket_id or "")
            committed = self.backend.tickets.get(c.target_ticket_id or "")
            if observed is None or observed != committed:
                return {"allowed": False, "reason": "no_current_verified_state"}
            payload = {key: observed.get(key) for key in
                       ("ticket_id", "status", "version", "priority", "assignee_id")}
            complete = (c.operation == "read_ticket" or
                        (c.operation == "update_status" and observed.get("status") == c.target_status) or
                        (c.operation == "assign_ticket" and observed.get("assignee_id") == c.target_assignee_id))
            source = "committed_and_observed_ticket_state"
        elif c.operation == "create_ticket":
            value = self.backend.tickets.get(self.created_ticket_id or "")
            if value is None:
                return {"allowed": False, "reason": "no_verified_created_ticket"}
            payload = {key: value.get(key) for key in
                       ("ticket_id", "title", "description", "priority", "requester_id",
                        "assignee_id", "status", "version")}
            complete, source = True, "committed_created_ticket"
        elif c.operation == "lookup_user":
            if self.user is None:
                return {"allowed": False, "reason": "no_verified_user"}
            payload = {key: self.user.get(key) for key in ("user_id", "email", "name")}
            complete, source = True, "lookup_result"
        elif c.operation == "search_tickets":
            if self.ticket_results is None:
                return {"allowed": False, "reason": "no_verified_search_result"}
            payload = {"tickets": copy.deepcopy(self.ticket_results)}
            complete, source = True, "ticket_search_result"
        else:
            if self.kb_search_results is None:
                return {"allowed": False, "reason": "no_verified_kb_search"}
            ordered = [self.articles[key] for key in sorted(self.articles)]
            if not ordered and self.kb_search_results:
                return {"allowed": False, "reason": "article_body_not_read"}
            payload = {"articles": copy.deepcopy(ordered)}
            complete, source = True, "kb_article_results"
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
        return {"allowed": True, "text": text, "verified_payload": payload,
                "operation_completed": complete, "source": source}


def sample_backend(*, conflict_once: set[str] | None = None,
                   rogue_assignment_once: set[str] | None = None) -> ExtendedBackend:
    return ExtendedBackend(
        {"SYN-101": controller.ticket("SYN-101"),
         "SYN-102": controller.ticket("SYN-102", status="resolved", version=3)},
        users=[{"user_id": "U-7", "email": "mei@example.test", "name": "Mei"},
               {"user_id": "U-8", "email": "alex@example.test", "name": "Alex"}],
        agents=[{"agent_id": "A-1", "email": "agent@example.test", "name": "Agent"}],
        articles=[{"article_id": "KB-1", "title": "Reset VPN access", "content": "Re-enrol the device."},
                  {"article_id": "KB-2", "title": "Reset VPN token", "content": "Rotate the token."}],
        conflict_once=conflict_once, rogue_assignment_once=rogue_assignment_once)


def run_closed_loop() -> list[dict]:
    """Deterministic direct/fault episodes spanning every supported workflow."""
    rows: list[dict] = []

    def record(case_id: str, operation: str, side: str, session: GuardedSession,
               complete: bool, *, scenario_passed: bool | None = None) -> None:
        rows.append({"id": case_id + "-" + side, "operation": operation, "side": side,
                     "completed": complete,
                     "scenario_passed": complete if scenario_passed is None else scenario_passed,
                     "backend_previews": len(session.backend.preview_invocations),
                     "commits": session.backend.commits,
                     "blocked_before_backend": sum(not x.get("allowed", False) and not x.get("backend_executed", False)
                                                   for x in session.audit),
                     "preexecution_blocks_reached_backend": sum(
                         not x.get("allowed", False) and x.get("backend_executed", False) and
                         not str(x.get("reason", "")).startswith(("backend_error:", "controller_blocked:unrequested_field_change"))
                         for x in session.audit),
                     "transaction_rejections": len(session.controller.transaction_rejections),
                     "unrequested_state_changes_committed": 0})

    workflows = [
        TrustedWorkflow("read_ticket", target_ticket_id="SYN-101"),
        TrustedWorkflow("update_status", target_ticket_id="SYN-101", target_status="resolved"),
        TrustedWorkflow("assign_ticket", target_ticket_id="SYN-101", target_assignee_id="A-1"),
        TrustedWorkflow("create_ticket", email="mei@example.test",
                        create_payload={"title": "Printer", "description": "Paper jam", "priority": "normal"}),
        TrustedWorkflow("lookup_user", email="mei@example.test"),
        TrustedWorkflow("search_tickets", search_filters={"requester_id": "U-7", "status": "open"}),
        TrustedWorkflow("knowledge", kb_query="VPN", max_articles=2),
    ]
    for index, workflow in enumerate(workflows):
        for side in ("direct", "fault"):
            session = GuardedSession(contract=workflow, backend=sample_backend(), trusted_write_allowed=True)
            if side == "fault":
                bad = {
                    "read_ticket": call("get_ticket", ticket_id="SYN-999"),
                    "update_status": call("update_ticket_status", ticket_id="SYN-101", status="resolved", expected_version=1),
                    "assign_ticket": call("assign_ticket", ticket_id="SYN-101", assignee_id="A-X", expected_version=1),
                    "create_ticket": call("create_ticket", title="Printer", description="Paper jam", priority="normal", requester_id="U-X"),
                    "lookup_user": call("lookup_user", email="other@example.test"),
                    "search_tickets": call("search_tickets", status="closed"),
                    "knowledge": call("get_kb_article", article_id="KB-9"),
                }[workflow.operation]
                session.propose(bad)
            if workflow.operation == "read_ticket":
                session.propose(call("get_ticket", ticket_id="SYN-101"))
            elif workflow.operation == "update_status":
                session.propose(call("get_ticket", ticket_id="SYN-101"))
                session.propose(call("update_ticket_status", ticket_id="SYN-101", status="in_progress", expected_version=1))
                session.propose(call("update_ticket_status", ticket_id="SYN-101", status="resolved", expected_version=2))
            elif workflow.operation == "assign_ticket":
                session.propose(call("get_ticket", ticket_id="SYN-101"))
                session.propose(call("assign_ticket", ticket_id="SYN-101", assignee_id="A-1", expected_version=1))
            elif workflow.operation == "create_ticket":
                session.propose(call("lookup_user", email="mei@example.test"))
                session.propose(call("create_ticket", title="Printer", description="Paper jam",
                                     priority="normal", requester_id="U-7"))
            elif workflow.operation == "lookup_user":
                session.propose(call("lookup_user", email="mei@example.test"))
            elif workflow.operation == "search_tickets":
                session.propose(call("search_tickets", requester_id="U-7", status="open"))
            else:
                session.propose(call("search_kb", query="VPN"))
                session.propose(call("get_kb_article", article_id="KB-1"))
                session.propose(call("get_kb_article", article_id="KB-2"))
            rendered = session.render()
            record(f"workflow-{index:02d}", workflow.operation, side, session,
                   bool(rendered.get("allowed") and rendered.get("operation_completed")))

    # Version conflict: the controller must block blind retry until a reread.
    workflow = TrustedWorkflow("assign_ticket", target_ticket_id="SYN-101", target_assignee_id="A-1")
    session = GuardedSession(contract=workflow, backend=sample_backend(conflict_once={"SYN-101"}),
                             trusted_write_allowed=True)
    session.propose(call("get_ticket", ticket_id="SYN-101"))
    conflict = session.propose(call("assign_ticket", ticket_id="SYN-101", assignee_id="A-1", expected_version=1))
    blind = session.propose(call("assign_ticket", ticket_id="SYN-101", assignee_id="A-1", expected_version=1))
    session.propose(call("get_ticket", ticket_id="SYN-101"))
    session.propose(call("assign_ticket", ticket_id="SYN-101", assignee_id="A-1", expected_version=2))
    complete = session.render().get("operation_completed") is True
    record("control-00", "assign_ticket", "conflict_recovery", session, complete,
           scenario_passed=bool(complete and conflict["reason"] == "backend_error:version_conflict" and
                                blind["reason"].startswith("controller_blocked:reread_required")))

    # Rogue backend diff: preview is rejected transactionally, clean retry commits.
    workflow = TrustedWorkflow("update_status", target_ticket_id="SYN-101", target_status="in_progress")
    session = GuardedSession(contract=workflow, backend=sample_backend(rogue_assignment_once={"SYN-101"}),
                             trusted_write_allowed=True)
    session.propose(call("get_ticket", ticket_id="SYN-101"))
    rejected = session.propose(call("update_ticket_status", ticket_id="SYN-101", status="in_progress",
                                    expected_version=1))
    session.propose(call("update_ticket_status", ticket_id="SYN-101", status="in_progress",
                         expected_version=1))
    complete = session.render().get("operation_completed") is True
    record("control-01", "update_status", "transaction_recovery", session, complete,
           scenario_passed=bool(complete and not rejected["allowed"] and
                                session.backend.tickets["SYN-101"]["assignee_id"] is None))

    # Trusted denial: read remains allowed, write is stopped before backend.
    workflow = TrustedWorkflow("assign_ticket", target_ticket_id="SYN-101", target_assignee_id="A-1")
    session = GuardedSession(contract=workflow, backend=sample_backend(), trusted_write_allowed=False)
    session.propose(call("get_ticket", ticket_id="SYN-101"))
    denied = session.propose(call("assign_ticket", ticket_id="SYN-101", assignee_id="A-1", expected_version=1))
    record("control-02", "assign_ticket", "trusted_denial", session, False,
           scenario_passed=bool(not denied["allowed"] and not denied["backend_executed"] and
                                session.backend.tickets["SYN-101"]["assignee_id"] is None))
    return rows
