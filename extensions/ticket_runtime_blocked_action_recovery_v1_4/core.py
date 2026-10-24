"""General blocked-action recovery layered over frozen recovery v1.3.

Revision 1.4 adds two generic, contract-preserving continuations:

* an out-of-contract KB query is replaced by the exact trusted query and the
  resulting article identifiers are returned to the model;
* a redundant same-target ticket reread during an assignment recovery is
  honored, then the already-authorized exact assignment is completed using the
  freshly verified version.

Both paths record the rejected model proposal and the controller recovery as
separate events.  No call outside the original request contract reaches the
backend.
"""

from __future__ import annotations

import copy

from extensions.ticket_runtime_blocked_action_recovery_v1_3 import core as prior
from extensions.ticket_runtime_contract_guard_integration_v1 import core as runtime_v1
from extensions.ticket_runtime_contract_guard_v1_1 import core as guard
from extensions.ticket_runtime_transaction_recovery_v1_2 import core as recovery_v1_2


REVISION = "1.4"
SOURCE = "blocked_action_recovery_runtime_v1.4"
KB_TRIGGER = "query_outside_trusted_contract"


def _required_call(episode: "GeneralBlockedActionRecoveryEpisode") -> dict | None:
    if episode.workflow.operation == "knowledge":
        return guard.call("search_kb", query=episode.workflow.kb_query)
    return prior._required_call(episode)


def _goal(episode: "GeneralBlockedActionRecoveryEpisode") -> dict:
    if episode.workflow.operation == "knowledge":
        return {"operation": "knowledge", "query": episode.workflow.kb_query,
                "max_articles": episode.workflow.max_articles}
    return prior._goal(episode)


def feedback(episode: "GeneralBlockedActionRecoveryEpisode", reason: str, *,
             retryable: bool) -> dict:
    contract = copy.deepcopy(episode.recovery or {})
    if contract:
        contract["required_next_call"] = _required_call(episode)
        contract["remaining_nonprogress_budget"] = max(
            0, recovery_v1_2.MAX_NONPROGRESS - episode.nonprogress_count)
    return {"ok": False, "error": {
        "code": "blocked_action_recovery_required" if retryable else "bounded_safe_stop",
        "reason": reason, "retryable": retryable, "recovery_contract": contract,
    }, "source": SOURCE}


class GeneralBlockedActionRecoveryEpisode(prior.BlockedActionRecoveryEpisode):
    def _activate_kb_recovery(self, reason: str) -> None:
        self.recovery = {
            "revision": REVISION,
            "kind": "blocked_action_recovery",
            "trigger_family": "wrong_kb_query",
            "trigger_reason": reason,
            "phase": "exact_retry_required",
            "authorized_goal": _goal(self),
            "forbidden": ["broaden_request_scope", "substitute_query",
                          "read_unreturned_article", "claim_completion_without_evidence"],
        }
        self.nonprogress_count = 0

    def _record_runtime_block(self, proposed: dict, provenance: str,
                              reason: str) -> tuple[dict, dict]:
        result, _ = super()._record_runtime_block(proposed, provenance, reason)
        response = feedback(self, reason, retryable=True)
        self.events[-1]["response"] = copy.deepcopy(response)
        return result, response

    def _controller_continue(self, proposed: dict, provenance: str,
                             exact: dict, reason: str) -> tuple[dict, dict]:
        """Reject/observe the model action, then execute only the frozen exact goal."""
        if proposed["name"] == "get_ticket":
            read_result, read_response = runtime_v1.RuntimeEpisode.propose(
                self, proposed, provenance)
            if not read_result.get("allowed"):
                return read_result, read_response
        else:
            self._record_runtime_block(proposed, provenance, reason)
        result, response = runtime_v1.RuntimeEpisode.propose(
            self, exact, "guard_recovery")
        runtime_v1.require(result.get("allowed"), "Exact controller recovery was rejected")
        self.recovery["phase"] = "complete"  # type: ignore[index]
        self.nonprogress_count = 0
        wrapped = copy.deepcopy(response)
        wrapped["controller_recovery_call"] = copy.deepcopy(exact)
        wrapped["controller_recovery_reason"] = reason
        wrapped["source"] = SOURCE
        self.events[-1]["response"] = copy.deepcopy(wrapped)
        return result, wrapped

    def propose(self, proposed: dict, provenance: str) -> tuple[dict, dict]:
        active = bool(self.recovery and self.recovery.get("kind") == "blocked_action_recovery" and
                      self.recovery.get("phase") != "complete")
        if active:
            exact = _required_call(self)
            runtime_v1.require(exact is not None, "Active recovery has no exact call")
            if proposed == exact:
                result, response = runtime_v1.RuntimeEpisode.propose(self, proposed, provenance)
                if result.get("allowed"):
                    self.recovery["phase"] = "complete"  # type: ignore[index]
                    self.nonprogress_count = 0
                return result, response
            if (self.workflow.operation == "knowledge" and proposed.get("name") == "search_kb"):
                return self._controller_continue(
                    proposed, provenance, exact, "replace_with_exact_trusted_kb_query")
            if (self.workflow.operation == "assign_ticket" and proposed.get("name") == "get_ticket" and
                    proposed.get("arguments") == {"ticket_id": self.workflow.target_ticket_id}):
                return self._controller_continue(
                    proposed, provenance, exact, "fresh_reread_then_exact_authorized_assignment")
            result, response = prior.BlockedActionRecoveryEpisode.propose(
                self, proposed, provenance)
            if self.events and self.events[-1].get("response"):
                response = feedback(self, str(result.get("reason") or
                                                "proposal_not_exact_trusted_recovery_call"),
                                    retryable=not self.safe_stop)
                self.events[-1]["response"] = copy.deepcopy(response)
            return result, response

        if self.recovery and self.recovery.get("phase") == "complete":
            return recovery_v1_2.RecoveryRuntimeEpisode.propose(self, proposed, provenance)

        result, response = prior.BlockedActionRecoveryEpisode.propose(self, proposed, provenance)
        reason = str(result.get("reason") or "")
        if reason == KB_TRIGGER:
            self._activate_kb_recovery(reason)
            response = feedback(self, reason, retryable=True)
            self.events[-1]["response"] = copy.deepcopy(response)
        elif self.recovery and self.recovery.get("kind") == "blocked_action_recovery":
            self.recovery["revision"] = REVISION
            response = feedback(self, reason, retryable=True)
            self.events[-1]["response"] = copy.deepcopy(response)
        return result, response

    def process(self, text: str, provenance: str = "model_proposal") -> dict:
        result = prior.BlockedActionRecoveryEpisode.process(self, text, provenance)
        if result.get("kind") == "calls":
            for observation in result.get("observations", []):
                recovery_call = (observation.get("response") or {}).get(
                    "controller_recovery_call")
                if recovery_call:
                    observation["call"] = copy.deepcopy(recovery_call)
        return result


def run_episode(task: dict, provider, *, variant: str, progress=None) -> dict:
    episode = GeneralBlockedActionRecoveryEpisode(task)
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
            progress({"id": task["id"], "variant": variant,
                      "rounds": copy.deepcopy(rounds), "events": copy.deepcopy(episode.events)})
    return {
        "id": task["id"], "variant": variant, "arm": "blocked_action_recovery_v1.4",
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

