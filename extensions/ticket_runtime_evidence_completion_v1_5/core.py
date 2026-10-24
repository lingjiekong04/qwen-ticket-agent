"""Fail-closed recovery for two generic tool-runtime defects.

The layer composes the frozen general recovery v1.4.  It may remove unsupported
arguments from an otherwise complete tool call, but it never invents a missing
required value.  The projected call still passes the frozen schema, request
contract and controller before reaching the in-memory backend.

For knowledge workflows, an accepted exact search is followed by deterministic
reads of the returned article identifiers (bounded by the trusted contract).
Article identifiers are never copied from model text.
"""

from __future__ import annotations

import copy

from extensions.ticket_runtime_blocked_action_recovery_v1_4 import core as prior
from extensions.ticket_runtime_contract_guard_integration_v1 import core as runtime_v1
from extensions.ticket_runtime_contract_guard_v1_1 import core as guard


REVISION = "1.5"
SOURCE = "trusted_schema_and_evidence_completion_v1.5"


def _schema_projection(proposed: dict) -> dict | None:
    """Drop only unsupported keys; never fill, coerce, or replace values."""
    if not isinstance(proposed, dict) or set(proposed) != {"name", "arguments"}:
        return None
    name, arguments = proposed["name"], proposed["arguments"]
    if name not in guard.TOOL_ARGS or not isinstance(arguments, dict):
        return None
    required, allowed = guard.TOOL_ARGS[name]
    keys = set(arguments)
    if not required <= keys or not (keys - allowed):
        return None
    projected = {"name": name,
                 "arguments": {key: copy.deepcopy(value)
                               for key, value in arguments.items() if key in allowed}}
    return projected if guard.validate_call(projected) is None else None


class EvidenceCompletionEpisode(prior.GeneralBlockedActionRecoveryEpisode):
    """Preserve the trusted goal while repairing only mechanically safe defects."""

    def _record_projection_block(self, proposed: dict, provenance: str,
                                 corrected: dict) -> None:
        before = copy.deepcopy(self.backend.tickets)
        response = {"ok": False, "error": {
            "code": "unsupported_arguments_removed",
            "reason": "invalid_tool_or_arguments",
            "retryable": True,
            "corrected_call": copy.deepcopy(corrected),
        }, "source": SOURCE}
        scope = runtime_v1.design.preflight(
            self.task["request_scope"], proposed,
            trusted_write_allowed=self.task["trusted_write_allowed"])
        self.events.append({
            "call": copy.deepcopy(proposed), "provenance": provenance,
            "schema_valid": False, "scope_allowed": scope["allowed"],
            "controller_decision": {"allowed": False, "action": "block",
                                    "reasons": ["invalid_tool_or_arguments"]},
            "reached_backend": False, "response": response,
            "state_before": before, "state_after": copy.deepcopy(self.backend.tickets),
        })

    def _complete_kb_evidence(self, search_response: dict) -> list[dict]:
        if self.workflow.operation != "knowledge" or not search_response.get("ok"):
            return []
        results = search_response.get("result") or []
        calls = []
        for item in results[: self.workflow.max_articles]:
            article_id = item.get("article_id") if isinstance(item, dict) else None
            if not isinstance(article_id, str) or not article_id:
                continue
            if article_id in self.guard.articles:
                continue
            exact = guard.call("get_kb_article", article_id=article_id)
            result, _ = runtime_v1.RuntimeEpisode.propose(self, exact, "guard_recovery")
            runtime_v1.require(result.get("allowed"),
                               "Trusted article completion was rejected")
            calls.append(exact)
        return calls

    def propose(self, proposed: dict, provenance: str) -> tuple[dict, dict]:
        corrected = _schema_projection(proposed)
        if corrected is not None:
            self._record_projection_block(proposed, provenance, corrected)
            result, response = prior.GeneralBlockedActionRecoveryEpisode.propose(
                self, corrected, "guard_recovery")
            wrapped = copy.deepcopy(response)
            wrapped["controller_recovery_call"] = copy.deepcopy(corrected)
            wrapped["controller_recovery_reason"] = "remove_unsupported_arguments"
            wrapped["source"] = SOURCE
            if result.get("allowed") and corrected["name"] == "search_kb":
                wrapped["controller_evidence_calls"] = self._complete_kb_evidence(response)
            return result, wrapped

        result, response = prior.GeneralBlockedActionRecoveryEpisode.propose(
            self, proposed, provenance)
        if result.get("allowed") and proposed.get("name") == "search_kb":
            wrapped = copy.deepcopy(response)
            wrapped["controller_evidence_calls"] = self._complete_kb_evidence(response)
            wrapped["source"] = SOURCE
            return result, wrapped
        return result, response


def run_episode(task: dict, provider, *, variant: str, progress=None) -> dict:
    episode = EvidenceCompletionEpisode(task)
    messages = [{"role": "user", "content": task["request"]}]
    rounds: list[dict] = []
    runtime_v1.seed_prefix(episode, messages)
    termination = "generation_budget_exceeded"
    for index in range(1, runtime_v1.MAX_GENERATIONS + 1):
        current = runtime_v1.payload(messages, task)
        generated = provider.generate(copy.deepcopy(current))
        rounds.append({"index": index, "input": current,
                       "prompt_info": provider.prompt_info(copy.deepcopy(current)),
                       "response": copy.deepcopy(generated)})
        processed = episode.process(generated["text"])
        messages.append({"role": "assistant", "content": generated["text"]})
        if processed.get("released"):
            termination = "safe_stop" if processed["kind"] == "safe_stop" else "final_response"
            break
        if processed["kind"] == "calls":
            for item in processed["observations"]:
                call = (item["response"].get("controller_recovery_call") or item["call"])
                runtime_v1.append_observation(messages, call, item["response"])
            if episode.guard.halted:
                termination = "guard_halted"
                break
        else:
            messages.append({"role": "user", "content": runtime_v1.canonical(processed["feedback"])})
        if progress:
            progress({"id": task["id"], "variant": variant,
                      "rounds": copy.deepcopy(rounds), "events": copy.deepcopy(episode.events)})
    return {
        "id": task["id"], "variant": variant, "arm": "evidence_completion_v1.5",
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

