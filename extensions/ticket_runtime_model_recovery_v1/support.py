from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from extensions.ticket_runtime_model_recovery_design_v1 import core as design


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPORT = ROOT / "reports/ticket-runtime-model-recovery-v1"
RUN = ROOT / "runs/ticket-runtime-model-recovery-v1-gpu0"
MODEL = Path("/konglingjie/sft_lora/models/Qwen2.5-7B-Instruct")
ADAPTER = Path("/konglingjie/sft_lora/lora_sft_toolcall_day1/runs/ticket-policy-retention-sft-v1.2-gpu2/train/adapter")
GPU = "0"
GPU_NAME = "NVIDIA A100-SXM4-80GB"
STACK = {"torch": "2.6.0+cu124", "transformers": "4.51.3", "peft": "0.15.2", "llamafactory": "0.9.3"}
AUTH_TEXT = "授权使用物理GPU0（A100）运行retention v1.2的48次模型恢复配对评测；不训练、不调用真实工具、不部署"
DIST = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
        "GROUP_RANK", "ROLE_RANK", "NODE_RANK")
PREFIXES = ("SLURM_", "ACCELERATE_", "PMIX_", "PMI_", "OMPI_", "TORCHELASTIC_")


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def no_framework() -> None:
    forbidden = {"torch", "transformers", "llamafactory"}
    require(not any(name.split(".")[0] in forbidden for name in sys.modules),
            "Model framework unexpectedly imported before CPU admission")


def clean_environment(parent: dict[str, str], *, gpu: bool) -> tuple[dict[str, str], list[str]]:
    env = dict(parent)
    removed = [key for key in env if key in DIST or key.startswith(PREFIXES)]
    for key in removed:
        env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES=GPU if gpu else "", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", FORCE_TORCHRUN="0", TOKENIZERS_PARALLELISM="false",
               PYTHONDONTWRITEBYTECODE="1")
    return env, sorted(removed)


def check_environment(env: dict[str, str], *, gpu: bool) -> None:
    require(env.get("CUDA_VISIBLE_DEVICES") == (GPU if gpu else ""), "Wrong GPU visibility")
    require(env.get("FORCE_TORCHRUN", "0") == "0", "Automatic distributed launch forbidden")
    require(all(env.get(key) == "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")),
            "Offline mode required")
    require(not any(key in DIST or key.startswith(PREFIXES) for key in env),
            "Distributed-launch environment remnants present")


def stack_gate() -> dict[str, str]:
    require(sys.version_info[:2] == (3, 11), "Use fixed Python 3.11")
    actual = {name: importlib.metadata.version(name) for name in STACK}
    require(actual == STACK, "Pinned inference stack changed")
    return actual


def source_gate() -> dict:
    binding = read(HERE / "bindings.json")
    require(binding.get("version") == "ticket-runtime-model-recovery-v1", "Wrong source binding")
    for relative, expected in binding["files"].items():
        path = ROOT / relative
        require(path.is_file() and sha(path) == expected, f"Source file changed: {relative}")
    return binding


def design_gate(server: bool) -> dict:
    result = design.evaluate(server=server)
    require(result["passed"] and result["tasks"] == 24 and result["planned_model_episodes"] == 48,
            "Frozen recovery design changed")
    require(result["tasks_sha256"] == "7450ec33b23cca008881eca3130ea04b0580ff734b4d0f7109088312b391d4b5"
            and result["protocol_sha256"] == "1326c54f1c0110f663ac6c01fd77e4e19757b884b2088a3df3c7c0e8496770f2",
            "Frozen task/protocol hash changed")
    return result


def verify_cpu(server: bool) -> dict:
    no_framework()
    source_gate()
    design_result = design_gate(server)
    delivery = read(REPORT / "delivery.json")
    require(delivery["physical_gpu"] == "0" and delivery["planned_model_episodes"] == 48,
            "Delivery experiment changed")
    require(delivery["training_authorized"] is False and delivery["live_tools"] is False
            and delivery["deployment_authorized"] is False, "Delivery exceeds authorized scope")
    return {
        "passed": True,
        "stage": "CPU_model_recovery_GPU0_admission",
        "tasks": 24,
        "arms": ["raw", "controlled"],
        "planned_model_episodes": 48,
        "retained_model": "ticket-policy-retention-v1.2-gpu2",
        "retained_adapter_check": design_result["retained_adapter_check"],
        "physical_gpu": "0",
        "gpu_name": GPU_NAME,
        "authorization_required_for_run": AUTH_TEXT,
        "training_authorized": False,
        "training_started": False,
        "gpu_inference_authorized": False,
        "gpu_inference_started": False,
        "live_tools_executed": 0,
        "deployment_authorized": False,
        "old_final_test_read": False,
    }


def model_args() -> dict:
    return {
        "model_name_or_path": str(MODEL),
        "adapter_name_or_path": str(ADAPTER),
        "create_new_adapter": False,
        "template": "qwen",
        "infer_backend": "huggingface",
        "finetuning_type": "lora",
        "trust_remote_code": False,
        "infer_dtype": "bfloat16",
        "flash_attn": "sdpa",
        "do_sample": False,
        "max_new_tokens": 512,
        "repetition_penalty": 1.0,
    }


def weight_gate() -> dict:
    require(MODEL.is_dir() and ADAPTER.is_dir(), "Base model or retained adapter missing")
    require(sha(ADAPTER / "adapter_model.safetensors") == design.ADAPTER_WEIGHTS_SHA256
            and sha(ADAPTER / "adapter_config.json") == design.ADAPTER_CONFIG_SHA256,
            "Retained adapter changed")
    adapter_config = read(ADAPTER / "adapter_config.json")
    require(adapter_config.get("r") == 8 and adapter_config.get("lora_alpha") == 16
            and adapter_config.get("lora_dropout") == 0.05
            and not adapter_config.get("use_dora") and not adapter_config.get("use_rslora"),
            "Unexpected retained adapter structure")
    proof = read(ROOT / "reports/ticket-decision-training-v1/initializer-run.json")["artifacts"]
    base = proof["base_weights_sha256"]
    for name, expected in base.items():
        require(Path(name).name == name and sha(MODEL / name) == expected, f"Base model changed: {name}")
    tokenizer = read(ROOT / "data/ticket_sft_v3_stage2_train_release/token-audit.json")["tokenizer_asset_sha256"]
    for name, expected in tokenizer.items():
        require(sha(MODEL / name) == expected, f"Tokenizer asset changed: {name}")
    return {"adapter_weights_sha256": design.ADAPTER_WEIGHTS_SHA256,
            "adapter_config_sha256": design.ADAPTER_CONFIG_SHA256,
            "base_weights_sha256": base, "tokenizer_asset_sha256": tokenizer}


def gpu_snapshot() -> dict:
    def query(arguments: list[str]) -> str:
        return subprocess.run(["nvidia-smi", *arguments], check=True, capture_output=True,
                              text=True, timeout=20).stdout
    return {
        "identity_raw": query(["--id=0", "--query-gpu=uuid,name,memory.total,memory.free",
                               "--format=csv,noheader,nounits"]),
        "processes_raw": query(["--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"]),
        "observed_at_utc": now(), "observer_pid": os.getpid(), "physical_gpu": "0",
    }


def gpu_from_snapshot(snapshot: dict) -> dict:
    require(snapshot.get("physical_gpu") == "0", "Wrong physical GPU snapshot")
    lines = [line.strip() for line in snapshot["identity_raw"].splitlines() if line.strip()]
    require(len(lines) == 1, "Cannot identify exactly physical GPU0")
    fields = [value.strip() for value in lines[0].split(",")]
    require(len(fields) == 4, "Malformed GPU0 identity")
    uuid, name, total, free = fields
    require(uuid.startswith("GPU-") and "A100" in name and float(total) >= 75000,
            "Require full A100 80GB physical GPU0")
    for line in snapshot["processes_raw"].splitlines():
        if not line.strip():
            continue
        process = [value.strip() for value in line.split(",")]
        require(len(process) >= 3 and process[0].startswith("GPU-") and process[1].isdigit(),
                "Malformed GPU process evidence")
        require(process[0] != uuid, "GPU0 occupied; preserve outputs and do not stop other work")
    require(float(free) >= 70000, "GPU0 has less than 70GB free")
    return {"selection": "0", "uuid": uuid, "name": name,
            "total_mib": float(total), "free_mib": float(free),
            "raw_snapshot": snapshot,
            "scope": "Idle at observation only; not continuous exclusive GPU ownership"}


def gpu_idle_gate() -> dict:
    return gpu_from_snapshot(gpu_snapshot())


def authorization_gate() -> dict:
    receipt = read(RUN / "authorization.json")
    require(receipt.get("user_message") == AUTH_TEXT and receipt.get("physical_gpu") == "0"
            and receipt.get("planned_model_episodes") == 48
            and receipt.get("training_authorized") is False
            and receipt.get("live_tools") is False and receipt.get("deployment_authorized") is False,
            "Missing or mismatched execution authorization")
    return receipt

