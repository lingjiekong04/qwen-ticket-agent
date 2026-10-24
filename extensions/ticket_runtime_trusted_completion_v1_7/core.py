"""Contract-preserving completion for the three remaining v5 failure families.

This layer composes frozen exact-create recovery v1.6. It adds only three
deterministic continuations whose complete values are already present in the
model-visible request:

* replace an in-contract-schema but filter-divergent ``search_tickets`` call
  with the exact trusted query/requester/status mapping;
* after a wrong-ticket status action is blocked, reread the exact trusted
  target and perform only the explicitly requested status transition using the
  freshly observed version;
* if a guard recovery completed the goal during the injected prefix, release
  the deterministic evidence answer before requesting another model token.

No model value is copied into a recovery call and no missing value is inferred.
"""

from __future__ import annotations

import copy

from extensions.ticket_runtime_contract_guard_integration_v1 import core as runtime_v1
from extensions.ticket_runtime_contract_guard_v1_1 import core as guard
from extensions.ticket_runtime_exact_create_recovery_v1_6 import core as prior


REVISION = "1.7"
SOURCE = "exact_trusted_completion_runtime_v1.7"


def _all_visible(task: dict, values: list[object]) -> bool:
    request = task.get("request", "")
    return all(isinstance(value, (str, int)) and str(value) in request for value in values)


def _exact_search(episode: "TrustedCompletionEpisode") -> dict | None:
    filters = copy.deepcopy(episode.workflow.search_filters or {})
    if (episode.workflow.operation != "search_tickets" or
            set(filters) != {"query", "requester_id", "status"} or
            not _all_visible(episode.task, list(filters.values()))):
        return None
    call = {"name": "search_tickets", "arguments": filters}
    return call if guard.validate_call(call) is None else None


def _exact_status_calls(episode: "TrustedCompletionEpisode") -> tuple[dict, dict] | None:
    workflow = episode.workflow
    if (workflow.operation != "update_status" or not episode.task.get("trusted_write_allowed") or
            not workflow.target_ticket_id or not workflow.target_status or
            not _all_visible(episode.task, [workflow.target_ticket_id, workflow.target_status])):
        return None
    state = episode.backend.tickets.get(workflow.target_ticket_id)
    if state is None:
        return None
    reread = guard.call("get_ticket", ticket_id=workflow.target_ticket_id)
    update = guard.call("update_ticket_status", ticket_id=workflow.target_ticket_id,
                        status=workflow.target_status, expected_version=state["version"])
    return reread, update


class TrustedCompletionEpisode(prior.ExactCreatePayloadRecoveryEpisode):
    def _recover_search(self, proposed: dict, provenance: str,
                        result: dict, response: dict) -> tuple[dict, dict]:
        exact = _exact_search(self)
        reason = str(result.get("reason") or "")
        if not (exact and proposed.get("name") == "search_tickets" and proposed != exact and
                reason in {"filters_outside_trusted_contract",
                           "proposal_not_exact_trusted_recovery_call"}):
            return result, response
        recovered, recovered_response = runtime_v1.RuntimeEpisode.propose(
            self, exact, "guard_recovery")
        runtime_v1.require(recovered.get("allowed"), "Exact trusted search recovery was rejected")
        wrapped = copy.deepcopy(recovered_response)
        wrapped.update({
            "controller_recovery_call": copy.deepcopy(exact),
            "controller_recovery_reason": "replace_with_exact_visible_search_filters",
            "rejected_model_call": copy.deepcopy(proposed),
            "source": SOURCE,
        })
        self.events[-1]["response"] = copy.deepcopy(wrapped)
        self.recovery = {
            "revision": REVISION, "kind": "exact_search_ticket_recovery",
            "trigger_reason": reason, "phase": "complete",
            "authorized_goal": copy.deepcopy(exact),
            "forbidden": ["copy_model_value", "infer_missing_value",
                          "change_identifier_case", "broaden_request_scope"],
        }
        self.nonprogress_count = 0
        return recovered, wrapped

    def _recover_status_target(self, proposed: dict, provenance: str,
                               result: dict, response: dict) -> tuple[dict, dict]:
        calls = _exact_status_calls(self)
        reason = str(result.get("reason") or "")
        if not (calls and reason == "target_outside_trusted_contract" and
                proposed.get("name") in {"get_ticket", "update_ticket_status"}):
            return result, response
        reread, update = calls
        read_result, read_response = runtime_v1.RuntimeEpisode.propose(
            self, reread, "guard_recovery")
        runtime_v1.require(read_result.get("allowed"), "Trusted-target reread was rejected")
        latest = (read_response.get("result") or {}).get("version")
        runtime_v1.require(isinstance(latest, int), "Trusted-target reread returned no version")
        update["arguments"]["expected_version"] = latest
        update_result, update_response = runtime_v1.RuntimeEpisode.propose(
            self, update, "guard_recovery")
        runtime_v1.require(update_result.get("allowed"), "Exact trusted status update was rejected")
        wrapped = copy.deepcopy(update_response)
        wrapped.update({
            "controller_recovery_calls": [copy.deepcopy(reread), copy.deepcopy(update)],
            "controller_recovery_call": copy.deepcopy(update),
            "controller_recovery_reason": "reread_exact_target_then_apply_exact_status",
            "controller_reread_response": copy.deepcopy(read_response),
            "rejected_model_call": copy.deepcopy(proposed),
            "source": SOURCE,
        })
        self.events[-1]["response"] = copy.deepcopy(wrapped)
        self.recovery = {
            "revision": REVISION, "kind": "exact_status_target_recovery",
            "trigger_reason": reason, "phase": "complete",
            "authorized_goal": {"operation": "update_status",
                                "ticket_id": self.workflow.target_ticket_id,
                                "target_status": self.workflow.target_status},
            "trusted_reread_call": copy.deepcopy(reread),
            "trusted_update_call": copy.deepcopy(update),
            "forbidden": ["copy_model_target", "infer_missing_target",
                          "change_unrequested_fields", "change_unrequested_ticket"],
        }
        self.nonprogress_count = 0
        return update_result, wrapped

    def propose(self, proposed: dict, provenance: str) -> tuple[dict, dict]:
        result, response = super().propose(proposed, provenance)
        result, response = self._recover_search(proposed, provenance, result, response)
        result, response = self._recover_status_target(proposed, provenance, result, response)
        return result, response


def _release_completed_prefix(episode: TrustedCompletionEpisode) -> dict | None:
    rendered = episode.guard.render()
    if not rendered.get("allowed") or not rendered.get("operation_completed"):
        return None
    released = episode.render_if_complete(provenance="guard_recovery")
    runtime_v1.require(released.get("released"), "Completed trusted prefix did not release")
    return released


def run_episode(task: dict, provider, *, variant: str, progress=None) -> dict:
    episode = TrustedCompletionEpisode(task)
    messages = [{"role": "user", "content": task["request"]}]
    rounds: list[dict] = []
    runtime_v1.seed_prefix(episode, messages)
    prefix_release = _release_completed_prefix(episode)
    termination = "final_response" if prefix_release else "generation_budget_exceeded"
    if prefix_release:
        messages.append({"role": "assistant", "content": prefix_release["text"]})
    else:
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
        "id": task["id"], "variant": variant, "arm": "trusted_completion_v1.7",
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
        "prefix_completion_release": bool(prefix_release),
        "scores": None, "oracle_loaded_during_generation": False,
    }


ScriptedProvider = runtime_v1.ScriptedProvider

