"""Immutable admission and source bindings for the one-shot v6 blind run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any

from extensions.ticket_runtime_exact_create_recovery_v1_6 import core as baseline
from extensions.ticket_runtime_trusted_completion_v1_7 import core as candidate
from extensions.ticket_runtime_contract_guard_integration_v1 import core as runtime_v1
from extensions.ticket_runtime_model_recovery_v1 import support as model_support


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/ticket_runtime_prospective_holdout_v6"
TASKS, ORACLE = DATA / "tasks.json", DATA / "oracle.json"
ADMISSION, ISOLATION, MANIFEST = (DATA / "admission.json", DATA / "isolation.json",
                                  DATA / "manifest.json")
RUN = ROOT / "runs/ticket-runtime-prospective-blind-v6-gpu0"
MODEL = Path("/konglingjie/sft_lora/models/Qwen2.5-7B-Instruct")
ADAPTER = ROOT / "runs/ticket-policy-retention-sft-v1.2-gpu2/train/adapter"
ARMS = ("exact_create_recovery_v1.6", "trusted_completion_v1.7")
AUTH_TEXT = ("授权使用物理GPU0（A100）按冻结的独立v6正式盲测执行包一次性运行 retention v1.2 的320次 "
             "exact create recovery v1.6/trusted completion v1.7 对照评测；不训练、不调用真实工具、不部署")
FILES_SHA256 = {
    "tasks.json": "3fc6ce390f28538a3d3d3d078e25093d5b4201bca8101093f9caedf5f43c2a94",
    "oracle.json": "faafa08a1abb52281c952b797b28cb696417157d5b3e7924af53d6be392907a5",
    "admission.json": "159a978dff23de216f9cadfa13c00fb73aa9446f4f7a4c256b2f2caa338519a6",
    "isolation.json": "a2fa5ebae4525f1a66444788eddc0434a5cb63f3077ed78d98533502223408d1",
    "manifest.json": "43844be7369b34cc5405ed796e68d6b27f04df0286ebb1efc343db946dcde96b",
}
RUNTIME_SOURCE_SHA256 = {
    "extensions/ticket_runtime_contract_guard_v1/core.py":
        "bc4f506dee8897389d37c6abf057585e2b5cda1bb2ffde78b5c1ce8fd2492be8",
    "extensions/ticket_runtime_controller_v1_1/core.py":
        "be3622b92553bc15550949f7e035e5e9a5679e1bf3c430fa153735cc1b6ee3f2",
    "extensions/ticket_runtime_controller_v1_1_design/core.py":
        "312b9ab22b46f2bae620ec2d197619d3f4ac7d2accc57296700da05a01cb4d3c",
    "extensions/ticket_runtime_contract_guard_v1_1/core.py":
        "db9630c246662baea0b3a81c342475142f88ddb0c4971681e8f61ba89c0f4260",
    "extensions/ticket_runtime_contract_guard_integration_v1/core.py":
        "bfde04bfcbd53e627f555a816de6d04e91315d57494fc65da9383ee6af8f00b3",
    "extensions/ticket_runtime_transaction_recovery_v1_2/core.py":
        "a025f27442c5a61014deb6f27a7f7eb342d2b364df6d8b9c2b33cfc0b213ece8",
    "extensions/ticket_runtime_blocked_action_recovery_v1_3/core.py":
        "70aae82403935920f0d9d21898094499e7c716cd9c8b634e34a21fdcc05126cf",
    "extensions/ticket_runtime_blocked_action_recovery_v1_4/core.py":
        "ceb56ac97e1a39694addcaa71cb5a6009f8698bf2e7272502bac207d8fbfeda3",
    "extensions/ticket_runtime_evidence_completion_v1_5/core.py":
        "ade7453442eed59862269257587eef3dfe19082248e8cb0ac19c3482401e783e",
    "extensions/ticket_runtime_exact_create_recovery_v1_6/core.py":
        "35e16bff1f9c763703bfab2cb07ecc694b9073ddf1cd24665879c9ac7733f580",
    "extensions/ticket_runtime_trusted_completion_v1_7/core.py":
        "aa225d09730513280b433e6ea5b45e97d0f9dc4f926095b5ea719247d7309043",
    "extensions/ticket_runtime_model_recovery_v1/support.py":
        "594be38be0f97344614704a7ad407ae6d60d20060884a592f037774a4a354430",
    "extensions/ticket_runtime_model_recovery_v1/provider.py":
        "9438a7a748be467d497cf78e9201682fd70e1f39d31f78238e63f8854ba2ce5f",
    "extensions/ticket_runtime_prospective_holdout_v6/core.py":
        "09ee835b23ea32928a00e382271274979059afbc6e36140f3a862cef7a1f9985",
    "extensions/ticket_runtime_prospective_holdout_v6/tests.py":
        "e76d5d6ee4b29925ebb0a4904c59cb5aa154df35c40b792acbaa5e224b6c50ee",
}
ADAPTER_SHA256 = "1837ddf22cce6a614c360ac86e0c1d48b261334db4712ac10b7810472e4cb625"
ADAPTER_CONFIG_SHA256 = "57835d0546a21af767e2c1acf8b8153bd2e4f01cfee65d40a9e1438754666610"
STACK = {"torch": "2.6.0+cu124", "transformers": "4.51.3",
         "peft": "0.15.2", "llamafactory": "0.9.3"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def no_model_framework() -> None:
    forbidden = {"torch", "transformers", "llamafactory", "peft"}
    require(not any(name.split(".")[0] in forbidden for name in sys.modules),
            "Model framework unexpectedly imported during CPU admission")


def protocol() -> dict:
    return {
        "kind": "one_shot_independent_all_tool_final_holdout_v6",
        "tasks": 160, "variant": "ticket-policy-retention-v1.2-gpu2",
        "runtime_arms": list(ARMS), "episodes_per_arm": 160,
        "planned_model_episodes": 320,
        "model": "Qwen2.5-7B-Instruct_plus_frozen_retention_v1.2_lora",
        "physical_gpu": "0", "gpu_name": "NVIDIA A100-SXM4-80GB",
        "generation": {"do_sample": False, "max_new_tokens": 512,
                       "repetition_penalty": 1.0, "template": "qwen",
                       "infer_dtype": "bfloat16", "flash_attn": "sdpa",
                       "episode_max_generations": runtime_v1.MAX_GENERATIONS,
                       "episode_max_tool_proposals": runtime_v1.MAX_PROPOSALS},
        "ordering": "tasks_in_frozen_order; each task comparison then candidate",
        "oracle_separation": "GPU worker reads tasks only; CPU scorer loads oracle after worker exit",
        "single_shot": True, "formal_admission": read(ADMISSION),
        "posthoc_review_can_change_formal_acceptance": False,
        "training": False, "live_tools": False, "deployment": False,
    }


def _validate_population(tasks: list[dict]) -> None:
    require(len(tasks) == 160 and len({task["id"] for task in tasks}) == 160,
            "Frozen v6 population changed")
    for tool in {task["primary_tool"] for task in tasks}:
        rows = [task for task in tasks if task["primary_tool"] == tool]
        require(len(rows) == 20 and
                sum(row["entry_kind"] == "direct_control" for row in rows) == 10 and
                sum(row["entry_kind"] == "forced_hazard_prefix" for row in rows) == 10,
                "Frozen v6 tool/stratum quota changed: " + tool)
        for task in rows:
            runtime_v1.workflow_from_task(task)


def worker_source_gate() -> list[dict]:
    no_model_framework()
    for name in ("tasks.json", "manifest.json", "isolation.json"):
        path = DATA / name
        require(path.is_file() and sha(path) == FILES_SHA256[name],
                "Frozen v6 worker source changed: " + name)
    for relative, expected in RUNTIME_SOURCE_SHA256.items():
        path = ROOT / relative
        require(path.is_file() and sha(path) == expected,
                "Frozen v6 runtime source changed: " + relative)
    manifest, isolation = read(MANIFEST), read(ISOLATION)
    require(manifest["holdout_acceptance_consumed"] is False and
            manifest["model_evaluated"] is False, "Frozen v6 manifest already consumed")
    require(isolation["exact_request_overlap"] == 0 and
            isolation["normalized_expression_overlap"] == 0 and
            isolation["entity_namespace_hits_prior_evaluation"] == 0 and
            isolation["entity_namespace_hits_training"] == 0 and
            isolation["prior_answers_or_scores_read"] is False and
            isolation["prior_oracles_read"] is False and
            isolation["development_and_v3_v4_v5_role"] ==
            "isolation_only_not_tuning_or_acceptance", "Frozen v6 isolation changed")
    tasks = read(TASKS)
    _validate_population(tasks)
    return tasks


def source_gate() -> tuple[list[dict], list[dict], dict]:
    tasks = worker_source_gate()
    for name in ("oracle.json", "admission.json"):
        require(sha(DATA / name) == FILES_SHA256[name], "Frozen v6 CPU source changed: " + name)
    oracle, admission = read(ORACLE), read(ADMISSION)
    require(len(oracle) == 160 and [row["task_id"] for row in oracle] ==
            [task["id"] for task in tasks], "Frozen v6 oracle population changed")
    require(all(row["task_sha256"] == runtime_v1.digest(task)
                for row, task in zip(oracle, tasks)), "Frozen v6 oracle binding changed")
    require(admission == protocol()["formal_admission"], "Frozen v6 admission changed")
    return tasks, oracle, admission


def weights_gate(server: bool) -> dict:
    if not server:
        return {"adapter_sha256": ADAPTER_SHA256, "server_checked": False}
    require(sys.version_info[:2] == (3, 11), "Pinned Python 3.11 required")
    require({name: importlib.metadata.version(name) for name in STACK} == STACK,
            "Pinned inference stack changed")
    model_support.weight_gate()
    require(sha(ADAPTER / "adapter_model.safetensors") == ADAPTER_SHA256 and
            sha(ADAPTER / "adapter_config.json") == ADAPTER_CONFIG_SHA256,
            "Frozen retention adapter changed")
    return {"adapter_sha256": ADAPTER_SHA256,
            "adapter_config_sha256": ADAPTER_CONFIG_SHA256, "server_checked": True}


def verify(server: bool = False) -> dict:
    tasks, _, admission = source_gate()
    require(not RUN.exists(), "Final v6 run path exists; preserve it and do not rerun")
    return {
        "passed": True, "stage": "CPU_one_shot_independent_final_holdout_v6_execution_foundation",
        "tasks": len(tasks), "runtime_arms": list(ARMS), "planned_model_episodes": 320,
        "protocol": protocol(), "protocol_sha256": runtime_v1.digest(protocol()),
        "tasks_sha256": FILES_SHA256["tasks.json"],
        "oracle_sha256": FILES_SHA256["oracle.json"],
        "admission_sha256": FILES_SHA256["admission.json"],
        "manifest_sha256": FILES_SHA256["manifest.json"],
        "weights": weights_gate(server), "formal_admission": admission,
        "run_path_absent": True, "execution_authorized": False,
        "gpu_inference_authorized": False,
        "authorization_required_for_execution": AUTH_TEXT,
        "holdout_acceptance_consumed": False, "model_loaded": False,
        "training_started": False, "gpu_inference_started": False,
        "live_tools_executed": 0, "deployment_authorized": False,
        "scope": "CPU-only immutable source, runtime, weight and one-shot v6 protocol admission",
    }


def model_args() -> dict:
    return {"model_name_or_path": str(MODEL), "adapter_name_or_path": str(ADAPTER),
            "create_new_adapter": False, "template": "qwen", "infer_backend": "huggingface",
            "finetuning_type": "lora", "trust_remote_code": False,
            "infer_dtype": "bfloat16", "flash_attn": "sdpa", "do_sample": False,
            "max_new_tokens": 512, "repetition_penalty": 1.0}


def runtime_for_arm(arm: str):
    require(arm in ARMS, "Unknown v6 runtime arm")
    return baseline if arm == ARMS[0] else candidate
