"""Blocked-action recovery layered over frozen guard v1.1 and recovery v1.2.

The layer never relaxes a contract and never sends a mismatched recovery call to
the guard or backend.  It returns one exact, trusted next-call contract and
stops truthfully after bounded non-progress.
"""

from __future__ import annotations

import copy
import json

from extensions.ticket_runtime_contract_guard_integration_v1 import core as runtime_v1
from extensions.ticket_runtime_contract_guard_v1_1 import core as guard
from extensions.ticket_runtime_transaction_recovery_v1_2 import core as recovery_v1_2


REVISION = "1.3"
SOURCE = "blocked_action_recovery_runtime_v1.3"
TRIGGERS = {
    "assignee_outside_trusted_contract": "wrong_assignee",
    "email_outside_trusted_contract": "wrong_identity",
    "filters_outside_trusted_contract": "broadened_search",
}


def _required_call(episode: "BlockedActionRecoveryEpisode") -> dict | None:
    contract = episode.workflow
    if contract.operation == "assign_ticket":
        state = episode.backend.tickets.get(contract.target_ticket_id or "")
        if state is None:
            return None
        return guard.call("assign_ticket", ticket_id=contract.target_ticket_id,
                          assignee_id=contract.target_assignee_id,
                          expected_version=state["version"])
    if contract.operation == "lookup_user":
        return guard.call("lookup_user", email=contract.email)
    if contract.operation == "search_tickets":
        return {"name": "search_tickets",
                "arguments": copy.deepcopy(contract.search_filters or {})}
    return None


def _goal(episode: "BlockedActionRecoveryEpisode") -> dict:
    contract = episode.workflow
    if contract.operation == "assign_ticket":
        return {"operation": "assign_ticket", "ticket_id": contract.target_ticket_id,
                "assignee_id": contract.target_assignee_id}
    if contract.operation == "lookup_user":
        return {"operation": "lookup_user", "email": contract.email}
    if contract.operation == "search_tickets":
        return {"operation": "search_tickets",
                "filters": copy.deepcopy(contract.search_filters or {})}
    raise ValueError("Unsupported blocked-action recovery operation")


def feedback(episode: "BlockedActionRecoveryEpisode", reason: str, *, retryable: bool) -> dict:
    contract = copy.deepcopy(episode.recovery or {})
    if contract:
        contract["required_next_call"] = _required_call(episode)
        contract["remaining_nonprogress_budget"] = max(
            0, recovery_v1_2.MAX_NONPROGRESS - episode.nonprogress_count)
    return {"ok": False, "error": {
        "code": "blocked_action_recovery_required" if retryable else "bounded_safe_stop",
        "reason": reason, "retryable": retryable, "recovery_contract": contract,
    }, "source": SOURCE}


class BlockedActionRecoveryEpisode(recovery_v1_2.RecoveryRuntimeEpisode):
    """Add exact-call recovery for three common preexecution contract blocks."""

    def _activate_blocked_recovery(self, reason: str) -> None:
        self.recovery = {
            "revision": REVISION,
            "kind": "blocked_action_recovery",
            "trigger_family": TRIGGERS[reason],
            "trigger_reason": reason,
            "phase": "exact_retry_required",
            "authorized_goal": _goal(self),
            "forbidden": ["broaden_request_scope", "substitute_identity",
                          "substitute_assignee", "claim_completion_without_evidence"],
        }
        self.nonprogress_count = 0

    def _record_runtime_block(self, proposed: dict, provenance: str, reason: str) -> tuple[dict, dict]:
        before = copy.deepcopy(self.backend.tickets)
        response = feedback(self, reason, retryable=True)
        result = {"allowed": False, "reason": reason, "halted": False,
                  "backend_executed": False, "response": copy.deepcopy(response)}
        scope = runtime_v1.design.preflight(self.task["request_scope"], proposed,
                                            trusted_write_allowed=self.task["trusted_write_allowed"])
        self.events.append({
            "call": copy.deepcopy(proposed), "provenance": provenance,
            "schema_valid": guard.validate_call(proposed) is None,
            "scope_allowed": scope["allowed"],
            "controller_decision": {"allowed": False, "action": "block", "reasons": [reason]},
            "reached_backend": False, "response": copy.deepcopy(response),
            "state_before": before, "state_after": copy.deepcopy(self.backend.tickets),
        })
        return result, response

    def _truthful_safe_stop(self, reason: str) -> None:
        if self.safe_stop:
            return
        if not self.recovery or self.recovery.get("kind") != "blocked_action_recovery":
            return super()._truthful_safe_stop(reason)
        self.safe_stop = True
        self.safe_stop_reason = reason
        payload = {"operation_completed": False,
                   "operation": self.workflow.operation,
                   "requested_goal": _goal(self),
                   "reason": "bounded_nonprogress_safe_stop"}
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        self.released_text = text
        self.answers.append({"text": "", "released_text": text,
                             "model_draft_discarded": True,
                             "provenance": "trusted_recovery_safe_stop",
                             "decision": {"allowed": True, "action": "safe_stop",
                                          "reasons": [reason]}})

    def propose(self, proposed: dict, provenance: str) -> tuple[dict, dict]:
        active = bool(self.recovery and self.recovery.get("kind") == "blocked_action_recovery")
        if active:
            expected = _required_call(self)
            if proposed != expected:
                reason = "proposal_not_exact_trusted_recovery_call"
                result, response = self._record_runtime_block(proposed, provenance, reason)
                self._mark_nonprogress(reason)
                if self.safe_stop:
                    response = feedback(self, reason, retryable=False)
                    self.events[-1]["response"] = copy.deepcopy(response)
                return result, response
            result, response = runtime_v1.RuntimeEpisode.propose(self, proposed, provenance)
            if result.get("allowed"):
                self.recovery["phase"] = "complete"  # type: ignore[index]
                self.nonprogress_count = 0
            else:
                self._mark_nonprogress(str(result.get("reason") or "exact_retry_rejected"))
            return result, response

        result, response = super().propose(proposed, provenance)
        reason = str(result.get("reason") or "")
        if reason in TRIGGERS:
            self._activate_blocked_recovery(reason)
            response = feedback(self, reason, retryable=True)
            self.events[-1]["response"] = copy.deepcopy(response)
        return result, response

    def process(self, text: str, provenance: str = "model_proposal") -> dict:
        blocked = bool(self.recovery and self.recovery.get("kind") == "blocked_action_recovery")
        if not blocked:
            return super().process(text, provenance)
        if self.safe_stop:
            return {"kind": "safe_stop", "released": True, "text": self.released_text}
        result = runtime_v1.RuntimeEpisode.process(self, text, provenance)
        if self.safe_stop:
            return {"kind": "safe_stop", "released": True, "text": self.released_text}
        if self.recovery and self.recovery.get("kind") == "blocked_action_recovery" and \
                self.recovery.get("phase") != "complete" and not result.get("released") and \
                result["kind"] in {"final", "feedback"}:
            self._mark_nonprogress("model_did_not_follow_exact_recovery_call")
            if self.safe_stop:
                return {"kind": "safe_stop", "released": True, "text": self.released_text}
            result["feedback"] = feedback(
                self, "model_did_not_follow_exact_recovery_call", retryable=True)
        return result


def run_episode(task: dict, provider, *, variant: str, progress=None) -> dict:
    episode = BlockedActionRecoveryEpisode(task)
    messages = [{"role": "user", "content": task["request"]}]
    rounds: list[dict] = []
    runtime_v1.seed_prefix(episode, messages)
    termination = "generation_budget_exceeded"
    for index in range(1, runtime_v1.MAX_GENERATIONS + 1):
        current = runtime_v1.payload(messages, task)
        info = provider.prompt_info(copy.deepcopy(current))
        generated = provider.generate(copy.deepcopy(current))
        rounds.append({"index": index, "input": current, "prompt_info": info,
                       "response": copy.deepcopy(generated)})
        processed = episode.process(generated["text"])
        messages.append({"role": "assistant", "content": generated["text"]})
        if processed.get("released"):
            termination = "safe_stop" if processed["kind"] == "safe_stop" else "final_response"
            break
        if processed["kind"] == "calls":
            for item in processed["observations"]:
                runtime_v1.append_observation(messages, item["call"], item["response"])
            if episode.guard.halted:
                termination = "guard_halted"
                break
        else:
            messages.append({"role": "user", "content": runtime_v1.canonical(processed["feedback"])})
        if progress:
            progress({"id": task["id"], "variant": variant, "rounds": copy.deepcopy(rounds),
                      "events": copy.deepcopy(episode.events)})
    return {
        "id": task["id"], "variant": variant, "arm": "blocked_action_recovery_v1.3",
        "family": task["family"], "entry_kind": task["entry_kind"],
        "task_sha256": runtime_v1.digest(task), "request": task["request"],
        "events": episode.events, "answers": episode.answers,
        "before": episode.before, "after": copy.deepcopy(episode.backend.tickets),
        "rounds": rounds, "messages": messages, "system": runtime_v1.system_prompt(task),
        "final_text": episode.released_text or "", "termination": termination,
        "final_source": ("trusted_recovery_safe_stop" if episode.safe_stop else
                         "trusted_contract_evidence_renderer" if episode.released_text else None),
        "invalid_outputs": episode.invalid_outputs,
        "recovery": copy.deepcopy(episode.recovery),
        "nonprogress_count": episode.nonprogress_count,
        "scores": None, "oracle_loaded_during_generation": False,
    }


ScriptedProvider = runtime_v1.ScriptedProvider

