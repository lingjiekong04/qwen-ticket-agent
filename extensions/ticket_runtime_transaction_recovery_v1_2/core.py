"""Generic transaction recovery layered over the frozen contract guard v1.1.

This module is model-provider agnostic and uses only in-memory backends in CPU
validation.  It does not alter the frozen controller, guard or v1 runtime.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from extensions.ticket_runtime_contract_guard_integration_v1 import core as prior
from extensions.ticket_runtime_controller_v1_1_design import core as design


REVISION = "1.2"
MAX_NONPROGRESS = 2
SOURCE = "transaction_recovery_runtime_v1.2"


def _ticket_state(episode: "RecoveryRuntimeEpisode") -> dict | None:
    ticket_id = episode.workflow.target_ticket_id or ""
    value = episode.backend.tickets.get(ticket_id)
    return copy.deepcopy(value) if value else None


def _next_call(episode: "RecoveryRuntimeEpisode", phase: str) -> dict | None:
    contract = episode.workflow
    ticket_id = contract.target_ticket_id
    state = _ticket_state(episode)
    if not ticket_id or state is None:
        return None
    if phase == "reread_required":
        return {"name": "get_ticket", "arguments": {"ticket_id": ticket_id}}
    if contract.operation == "update_status":
        next_status = design.ALLOWED_NEXT.get(state.get("status"))
        if next_status is None:
            return None
        return {"name": "update_ticket_status", "arguments": {
            "ticket_id": ticket_id,
            "status": next_status,
            "expected_version": state.get("version"),
        }}
    if contract.operation == "assign_ticket":
        return {"name": "assign_ticket", "arguments": {
            "ticket_id": ticket_id,
            "assignee_id": contract.target_assignee_id,
            "expected_version": state.get("version"),
        }}
    return None


def structured_feedback(episode: "RecoveryRuntimeEpisode", reason: str, *,
                        retryable: bool = True) -> dict:
    recovery = copy.deepcopy(episode.recovery or {})
    if recovery:
        recovery["current_verified_state"] = _ticket_state(episode)
        recovery["required_next_call"] = _next_call(episode, recovery["phase"])
        recovery["remaining_nonprogress_budget"] = max(
            0, MAX_NONPROGRESS - episode.nonprogress_count)
    return {
        "ok": False,
        "error": {
            "code": "transaction_recovery_required" if retryable else "bounded_safe_stop",
            "reason": reason,
            "retryable": retryable,
            "recovery_contract": recovery,
        },
        "source": SOURCE,
    }


class RecoveryRuntimeEpisode(prior.RuntimeEpisode):
    """Runtime with explicit rollback recovery and bounded non-progress."""

    def __init__(self, task: dict) -> None:
        super().__init__(task)
        self.recovery: dict | None = None
        self.nonprogress_count = 0
        self.safe_stop = False
        self.safe_stop_reason: str | None = None

    def _activate_recovery(self, reason: str, before: dict, after: dict) -> None:
        contract = self.workflow
        prior.require(contract.operation in {"update_status", "assign_ticket"},
                      "Transactional recovery is only valid for ticket writes")
        prior.require(before == after, "Rejected transaction changed committed state")
        state = _ticket_state(self)
        prior.require(state is not None, "Recovery target state is missing")
        goal: dict[str, Any] = {"operation": contract.operation,
                                "ticket_id": contract.target_ticket_id}
        if contract.operation == "update_status":
            goal["target_status"] = contract.target_status
        else:
            goal["target_assignee_id"] = contract.target_assignee_id
        self.recovery = {
            "revision": REVISION,
            "kind": "transaction_rollback_recovery",
            "rollback_confirmed": True,
            "phase": "reread_required",
            "authorized_goal": goal,
            "trigger_reason": reason,
            "forbidden": [
                "change_unrequested_fields",
                "change_unrequested_targets",
                "claim_completion_before_verified_goal",
            ],
        }
        self.nonprogress_count = 0

    def _truthful_safe_stop(self, reason: str) -> None:
        if self.safe_stop:
            return
        self.safe_stop = True
        self.safe_stop_reason = reason
        state = _ticket_state(self) or {}
        locale = self.task.get("locale", "en")
        target = (self.workflow.target_status if self.workflow.operation == "update_status"
                  else self.workflow.target_assignee_id)
        payload = {
            "operation_completed": False,
            "ticket_id": state.get("ticket_id"),
            "status": state.get("status"),
            "version": state.get("version"),
            "priority": state.get("priority"),
            "assignee_id": state.get("assignee_id"),
            "requested_target": target,
            "reason": "bounded_nonprogress_safe_stop",
        }
        profile = self.task.get("profile", "json")
        if profile == "json":
            text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        elif locale == "zh":
            assignee = state.get("assignee_id") or "未分配"
            text = (f"操作未完成并已安全停止；工单ID={state.get('ticket_id')}；"
                    f"当前状态={state.get('status')}；版本={state.get('version')}；"
                    f"优先级={state.get('priority')}；负责人={assignee}；请求目标={target}。")
        else:
            assignee = state.get("assignee_id") or "unassigned"
            text = (f"The operation was not completed and stopped safely; "
                    f"ticket_id={state.get('ticket_id')}; current_status={state.get('status')}; "
                    f"version={state.get('version')}; priority={state.get('priority')}; "
                    f"assignee_id={assignee}; requested_target={target}.")
        self.released_text = text
        self.answers.append({
            "text": "",
            "released_text": text,
            "model_draft_discarded": True,
            "provenance": "trusted_recovery_safe_stop",
            "decision": {"allowed": True, "action": "safe_stop", "reasons": [reason]},
        })

    def _mark_nonprogress(self, reason: str) -> None:
        self.nonprogress_count += 1
        if self.nonprogress_count >= MAX_NONPROGRESS:
            self._truthful_safe_stop(reason)

    def propose(self, proposed: dict, provenance: str) -> tuple[dict, dict]:
        recovery_before = copy.deepcopy(self.recovery)
        expected = (_next_call(self, recovery_before["phase"])
                    if recovery_before else None)
        before = copy.deepcopy(self.backend.tickets)
        result, response = super().propose(proposed, provenance)
        reason = str(result.get("reason", ""))

        if reason.startswith("controller_blocked:unrequested_field_change"):
            self._activate_recovery(reason, before, copy.deepcopy(self.backend.tickets))
            response = structured_feedback(self, reason)
            self.events[-1]["response"] = copy.deepcopy(response)
            return result, response

        if not recovery_before:
            return result, response

        if result.get("allowed") and proposed == expected:
            if recovery_before["phase"] == "reread_required":
                self.recovery["phase"] = "retry_required"  # type: ignore[index]
            else:
                rendered = self.guard.render()
                if rendered.get("operation_completed"):
                    self.recovery["phase"] = "complete"  # type: ignore[index]
                else:
                    self.recovery["phase"] = "retry_required"  # type: ignore[index]
            self.nonprogress_count = 0
            if self.recovery and self.recovery["phase"] != "complete":
                response = copy.deepcopy(response)
                response["recovery_contract"] = copy.deepcopy(
                    structured_feedback(self, "recovery_progress")["error"]["recovery_contract"])
                response["source"] = SOURCE
                self.events[-1]["response"] = copy.deepcopy(response)
            return result, response

        self._mark_nonprogress(reason or "proposal_not_required_next_action")
        response = structured_feedback(
            self, reason or "proposal_not_required_next_action", retryable=not self.safe_stop)
        self.events[-1]["response"] = copy.deepcopy(response)
        return result, response

    def process(self, text: str, provenance: str = "model_proposal") -> dict:
        if self.safe_stop:
            return {"kind": "safe_stop", "released": True, "text": self.released_text}
        result = super().process(text, provenance)
        if self.safe_stop:
            return {"kind": "safe_stop", "released": True, "text": self.released_text}
        if self.recovery and self.recovery.get("phase") != "complete" and not result.get("released"):
            if result["kind"] in {"final", "feedback"}:
                self._mark_nonprogress("model_did_not_follow_recovery_contract")
                if self.safe_stop:
                    return {"kind": "safe_stop", "released": True, "text": self.released_text}
                result["feedback"] = structured_feedback(
                    self, "model_did_not_follow_recovery_contract")
        return result


def run_episode(task: dict, provider, *, variant: str, progress=None) -> dict:
    episode = RecoveryRuntimeEpisode(task)
    messages = [{"role": "user", "content": task["request"]}]
    rounds: list[dict] = []
    prior.seed_prefix(episode, messages)
    termination = "generation_budget_exceeded"
    for index in range(1, prior.MAX_GENERATIONS + 1):
        current = prior.payload(messages, task)
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
                prior.append_observation(messages, item["call"], item["response"])
            if episode.guard.halted:
                termination = "guard_halted"
                break
        else:
            messages.append({"role": "user", "content": prior.canonical(processed["feedback"])})
        if progress:
            progress({"id": task["id"], "variant": variant, "rounds": copy.deepcopy(rounds),
                      "events": copy.deepcopy(episode.events)})
    return {
        "id": task["id"], "variant": variant, "arm": "transaction_recovery_v1.2",
        "family": task["family"], "entry_kind": task["entry_kind"],
        "task_sha256": prior.digest(task), "request": task["request"],
        "events": episode.events, "answers": episode.answers,
        "before": episode.before, "after": copy.deepcopy(episode.backend.tickets),
        "rounds": rounds, "messages": messages, "system": prior.system_prompt(task),
        "final_text": episode.released_text or "", "termination": termination,
        "final_source": ("trusted_recovery_safe_stop" if episode.safe_stop else
                         "trusted_contract_evidence_renderer" if episode.released_text else None),
        "invalid_outputs": episode.invalid_outputs,
        "transaction_recovery": copy.deepcopy(episode.recovery),
        "nonprogress_count": episode.nonprogress_count,
        "scores": None, "oracle_loaded_during_generation": False,
    }


ScriptedProvider = prior.ScriptedProvider
