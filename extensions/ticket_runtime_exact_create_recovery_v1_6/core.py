"""Fail-closed recovery for a schema-valid create call outside its trusted payload.

The original proposal is always blocked before backend execution. Recovery is
allowed only for a direct create workflow whose complete title, description,
priority and requester_id are present in the model-visible request. The runtime
then executes exactly those frozen values; it never copies or repairs model text.
"""

from __future__ import annotations

import copy

from extensions.ticket_runtime_contract_guard_integration_v1 import core as runtime_v1
from extensions.ticket_runtime_contract_guard_v1_1 import core as guard
from extensions.ticket_runtime_evidence_completion_v1_5 import core as prior


REVISION = "1.6"
SOURCE = "exact_trusted_create_payload_recovery_v1.6"
TRIGGER = "create_payload_outside_trusted_contract"


def _exact_create_call(episode: "ExactCreatePayloadRecoveryEpisode") -> dict | None:
    workflow = episode.workflow
    if (workflow.operation != "create_ticket" or workflow.requester_id is None or
            not episode.task.get("trusted_write_allowed")):
        return None
    payload = copy.deepcopy(workflow.create_payload or {})
    if set(payload) != {"title", "description", "priority"}:
        return None
    call = guard.call("create_ticket", requester_id=workflow.requester_id, **payload)
    request = episode.task.get("request", "")
    values = [payload["title"], payload["description"], payload["priority"],
              workflow.requester_id]
    if not all(isinstance(value, str) and value and value in request for value in values):
        return None
    return call


class ExactCreatePayloadRecoveryEpisode(prior.EvidenceCompletionEpisode):
    def propose(self, proposed: dict, provenance: str) -> tuple[dict, dict]:
        result, response = super().propose(proposed, provenance)
        reason = str(result.get("reason") or "")
        exact = _exact_create_call(self)
        if not (reason == TRIGGER and proposed.get("name") == "create_ticket" and
                guard.validate_call(proposed) is None and exact is not None and proposed != exact):
            return result, response
        recovered, recovered_response = runtime_v1.RuntimeEpisode.propose(
            self, exact, "guard_recovery")
        runtime_v1.require(recovered.get("allowed"),
                           "Exact trusted create recovery was rejected")
        wrapped = copy.deepcopy(recovered_response)
        wrapped["controller_recovery_call"] = copy.deepcopy(exact)
        wrapped["controller_recovery_reason"] = "replace_with_exact_trusted_create_payload"
        wrapped["rejected_model_call"] = copy.deepcopy(proposed)
        wrapped["source"] = SOURCE
        self.events[-1]["response"] = copy.deepcopy(wrapped)
        self.recovery = {
            "revision": REVISION, "kind": "exact_create_payload_recovery",
            "trigger_reason": TRIGGER, "phase": "complete",
            "authorized_goal": copy.deepcopy(exact),
            "forbidden": ["copy_model_value", "infer_missing_value",
                          "broaden_request_scope", "claim_without_backend_evidence"],
        }
        return recovered, wrapped


def run_episode(task: dict, provider, *, variant: str, progress=None) -> dict:
    episode = ExactCreatePayloadRecoveryEpisode(task)
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
                call = item["response"].get("controller_recovery_call") or item["call"]
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
        "id": task["id"], "variant": variant, "arm": "exact_create_recovery_v1.6",
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

