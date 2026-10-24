"""CPU-safe provenance, configuration, sampling and GPU admission checks."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / "runs/ticket-sft-v3-hermes-r8-v1"
RELEASE = ROOT / "data/ticket_sft_v3_stage2_train_release"
CANDIDATE = ROOT / "data/ticket_sft_v3_candidate_stage2"
DEV = ROOT / "data/ticket_agent_v2_dev"
MODEL = "/konglingjie/sft_lora/models/Qwen2.5-7B-Instruct"
OLD = "/konglingjie/sft_lora/lora_sft_toolcall_day1/runs/hermes-calls-r8-v1"
RELEASE_SHA = "986b2d6158e69de567c06faa0798d9f7a6b4739b6e6de2c6e7d2676cf33fd248"
DEV_SHA = "62fe2e50a21b398d9221d7f27cea79dace45e2a2976b5b9d83b2eb7e2998c01d"
OLD_SHA = "356142f89f4d95b2b8fde22d7a3c519b946b2d0328e3d4d3d5d5f2ba086a6bbb"
STACK = {"torch": "2.6.0+cu124", "transformers": "4.51.3", "peft": "0.15.2", "llamafactory": "0.9.3"}
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
QUOTAS = {"clarify": 500, "deny_write": 500, "no_tool": 500, "recovery": 400,
          "faithful": 400, "create": 300, "status": 200, "query": 200}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    def unique(items):
        obj = {}
        for k, v in items:
            require(k not in obj, "Duplicate JSON key")
            obj[k] = v
        return obj
    def bad(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=bad)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def update(path, value):
    # Only an owned fresh run's status file may be changed.
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat()


def fresh(path):
    require(not Path(path).exists(), f"Output exists; preserve it, no overwrite/resume: {path}")


def local_sources():
    files = sorted(HERE.rglob("*.py")) + sorted((ROOT / "configs").glob("ticket_sft_v3_*_v1.yaml"))
    files += [ROOT / "scripts/run_ticket_sft_v3.py"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files}


def delivery_gate():
    inventory = read(ROOT / "reports/ticket-training-v1-delivery.json")
    require(inventory["release_manifest_sha256"] == RELEASE_SHA and inventory["gpu"] == "1",
            "Wrong training delivery")
    for name, expected in inventory["files_sha256"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe delivery path")
        require(sha(ROOT / relative) == expected, f"Delivery file differs: {name}")


def legacy():
    # Use distinct package names for this extension; predecessor bare imports remain intact.
    directory = ROOT / "extensions/ticket_agent_v2"
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    import evaluate_agent
    return evaluate_agent


def check_adapter(config):
    require(config.get("r") == 8 and config.get("lora_alpha") == 16
            and config.get("lora_dropout") == 0.05 and set(config.get("target_modules", [])) == TARGETS,
            "Unexpected adapter rank/alpha/dropout/targets")
    for k in ("use_rslora", "use_dora", "rank_pattern", "alpha_pattern", "modules_to_save", "use_qalora"):
        require(not config.get(k), f"Unsupported adapter feature: {k}")
    require(config.get("bias", "none") == "none" and config.get("peft_type") == "LORA"
            and config.get("task_type") == "CAUSAL_LM", "Only classical causal-LM LoRA is permitted")


def config_for(stage):
    require(stage in ("preflight", "train"), "Unknown training stage")
    c = dict(model_name_or_path=MODEL, adapter_name_or_path=OLD, create_new_adapter=False,
             trust_remote_code=False, flash_attn="sdpa", stage="sft", do_train=True, do_eval=False,
             finetuning_type="lora", lora_rank=8, lora_alpha=16, lora_dropout=0.05, lora_target="all",
             use_dora=False, use_rslora=False,
             dataset_dir="data/ticket_sft_v3_stage2_train_release", dataset="ticket_sft_v3_train",
             template="qwen", cutoff_len=4096, train_on_prompt=False, mask_history=False, packing=False,
             preprocessing_num_workers=1, dataloader_num_workers=0, overwrite_cache=False,
             output_dir=f"runs/ticket-sft-v3-hermes-r8-v1/{stage}/adapter", overwrite_output_dir=False,
             logging_steps=10, logging_nan_inf_filter=False, save_strategy="epoch", save_total_limit=2,
             save_only_model=False, plot_loss=True, report_to="none", per_device_train_batch_size=1,
             gradient_accumulation_steps=8, learning_rate=2e-5, num_train_epochs=2.0,
             lr_scheduler_type="cosine", warmup_ratio=0.05, optim="adamw_torch",
             adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, weight_decay=0.0, max_grad_norm=1.0,
             bf16=True, disable_gradient_checkpointing=False, skip_memory_metrics=False,
             eval_strategy="no", load_best_model_at_end=False, seed=42, data_seed=42,
             push_to_hub=False, dataloader_drop_last=False)
    if stage == "preflight":
        c.update(dataset_dir="runs/ticket-sft-v3-hermes-r8-v1/preflight/data", dataset="ticket_sft_v3_preflight",
                 logging_steps=1, save_strategy="steps", save_steps=10, max_steps=10,
                 num_train_epochs=1.0, lr_scheduler_type="constant", warmup_ratio=0.0)
    return c


def check_config(config, stage):
    # Exact whitelist prevents extra flags, hidden mixing, resume or eval_dataset.
    require(config == config_for(stage), "Training configuration differs from the frozen recipe")


def select_preflight(episodes, records):
    require(len(episodes) == len(records) == 3000, "Incomplete source population")
    lengths = {r["id"]: r["total_tokens"] for r in records}
    grouped = defaultdict(lambda: defaultdict(list))
    for e in episodes:
        grouped[e["spec"]["category"]][e["pair_id"]].append(e)
    require(set(grouped) == set(QUOTAS), "Preflight categories differ")
    chosen = set()
    rng = random.Random(42)
    longest = {}
    for category in QUOTAS:
        pairs = grouped[category]
        require(all(len(v) == 2 and {e["locale"] for e in v} == {"zh", "en"} for v in pairs.values()),
                "Incomplete bilingual source pair")
        order = sorted(pairs, key=lambda p: (-max(lengths[e["id"]] for e in pairs[p]), p))
        longest[category] = order[0]
        rest = sorted(set(order) - {order[0]})
        chosen.update([order[0]] + rng.sample(rest, 4))
    indices = [i for i, e in enumerate(episodes) if e["pair_id"] in chosen]
    require(len(indices) == 80 and len(chosen) == 40
            and Counter(episodes[i]["spec"]["category"] for i in indices) == {k: 10 for k in QUOTAS},
            "Preflight sampling mismatch")
    return indices, longest


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def check_state(state, metrics, steps, epoch):
    require(state.get("global_step") == state.get("max_steps") == steps, "Planned optimizer steps not completed")
    require(finite(state.get("epoch")) and abs(state["epoch"] - epoch) < 1e-6, "Planned epochs not completed")
    require(finite(metrics.get("train_loss")), "Missing/nonfinite train_loss")
    for row in state.get("log_history", []):
        for key in ("loss", "train_loss", "grad_norm", "learning_rate"):
            if key in row:
                require(finite(row[key]), f"Nonfinite training metric: {key}")


def check_gpu_rows(identity_text, processes_text):
    rows = [r.strip() for r in identity_text.splitlines() if r.strip()]
    require(len(rows) == 1, "Cannot identify exactly GPU 1")
    uuid, name, total, free_mem = [s.strip() for s in rows[0].split(",")]
    require(uuid.startswith("GPU-") and "A100" in name and float(total) >= 75000,
            "This recipe requires the allocated full A100 80GB GPU 1 (no MIG)")
    require(float(free_mem) >= 60000, "Less than 60GB free; stop instead of changing the recipe")
    for row in processes_text.splitlines():
        if row.strip():
            fields = [v.strip() for v in row.split(",")]
            require(len(fields) == 3, "Cannot reliably inspect compute processes")
            require(fields[0] != uuid, "GPU 1 has another compute process; do not kill or share it")
    return {"selection": "1", "uuid": uuid, "name": name, "total_mib": float(total), "free_mib": float(free_mem)}


def check_actual_gpu_uuid(expected, actual):
    require(actual is not None and str(actual).removeprefix("GPU-").lower()
            == expected.removeprefix("GPU-").lower(), "CUDA logical device differs from physical GPU 1 UUID")


def runtime_gate():
    require(sys.version_info[:2] == (3, 11), "Activate pinned remote Python 3.11")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and os.environ.get("WORLD_SIZE", "1") == "1"
            and os.environ.get("LOCAL_RANK", "-1") == "-1" and os.environ.get("FORCE_TORCHRUN", "0") == "0",
            "Require GPU 1 and non-distributed execution")
    require(os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1", "Require offline execution")
    versions = {k: importlib.metadata.version(k) for k in STACK}
    require(versions == STACK, f"Pinned stack differs: {versions}")
    identity = subprocess.run(["nvidia-smi", "--id=1", "--query-gpu=uuid,name,memory.total,memory.free",
                               "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=20).stdout
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                                "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=20).stdout
    gpu = check_gpu_rows(identity, processes)
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(),
            "Require exactly one BF16 CUDA GPU")
    check_actual_gpu_uuid(gpu["uuid"], getattr(torch.cuda.get_device_properties(0), "uuid", None))
    return versions, gpu


def release_gate():
    require(sha(RELEASE / "manifest.json") == RELEASE_SHA, "Wrong published release manifest")
    sys.path.insert(0, str(ROOT / "scripts"))
    from verify_ticket_sft_stage2_release import verify
    result = verify(RELEASE, CANDIDATE, ROOT / "reports/ticket-sft-v3-stage2-token-audit.json")
    require(result["passed"] and result["rows"] == 3000, "Release verification failed")
    return result


def artifact_gate():
    old_eval = legacy()
    choice = old_eval.verify_model_artifacts()
    require(choice["selected_model_path"] == MODEL and choice["selected_adapter_path"] == OLD,
            "Frozen model paths differ")
    require(sha(Path(OLD) / "adapter_model.safetensors") == OLD_SHA, "Wrong initializer")
    check_adapter(read(Path(OLD) / "adapter_config.json"))
    weights = old_eval.base_weight_hashes(Path(MODEL))
    reference = read(ROOT / "runs/ticket-agent-v2/dev/hermes-r8/run.json")
    require(reference["status"] == "complete" and reference["provider_kind"] == "huggingface"
            and reference["adapter_sha256"] == OLD_SHA and reference["base_weights_sha256"] == weights,
            "Base weights differ from prior real DEV provenance")
    report = read(RELEASE / "token-audit.json")
    for name, expected in report["tokenizer_asset_sha256"].items():
        require(sha(Path(MODEL) / name) == expected, f"Tokenizer/config differs: {name}")
    import inspect
    from llamafactory.data.converter import SharegptDatasetConverter
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    for cls in (SharegptDatasetConverter, SupervisedDatasetProcessor):
        require(sha(inspect.getsourcefile(cls)) == report["framework_source_sha256"][cls.__name__],
                "Training converter/processor differs from actual token audit")
    from llamafactory.hparams import get_train_args
    from llamafactory.train.sft.workflow import run_sft
    from llamafactory.model.adapter import init_adapter
    from transformers import Trainer
    framework = {obj.__name__: sha(inspect.getsourcefile(obj))
                 for obj in (get_train_args, run_sft, init_adapter, Trainer)}
    return {"choice_sha256": old_eval.CHOICE_SHA, "initializer_sha256": OLD_SHA,
            "initializer_config_sha256": sha(Path(OLD) / "adapter_config.json"), "base_weights_sha256": weights,
            "framework_source_sha256": framework}


def dev_gate():
    legacy()
    from prepare_dev import load_corpus
    from review_tasks import validate_approvals
    from tasks import build_tasks
    tasks, refs, manifest = load_corpus(DEV)
    from validate import frozen_dev
    require(tasks == build_tasks() and len(tasks) == 60 and digest(frozen_dev()) == DEV_SHA, "Frozen 60 DEV tasks changed")
    approvals = validate_approvals(DEV, read(DEV / "review-checklist.json"))
    return tasks, refs, {"corpus_manifest_sha256": sha(DEV / "manifest.json"), "frozen_tasks_sha256": DEV_SHA,
                          "approval_sha256": approvals, "legacy_source_sha256": legacy().source_hashes()}


def compare_tensor_sets(saved, loaded):
    # PeftModel names insert the adapter name; save_pretrained strips it.
    normalized = {k.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B."): v
                  for k, v in loaded.items()}
    require(len(normalized) == len(loaded) and set(saved) == set(normalized) and saved,
            "Loaded trainable tensor keys differ from saved LoRA")
    import torch
    for k, original in saved.items():
        actual = normalized[k].detach().cpu()
        require(actual.shape == original.shape and torch.equal(actual, original.to(dtype=actual.dtype)),
                f"Loaded initial/reloaded tensor differs: {k}")
        require(torch.isfinite(actual).all().item(), f"Nonfinite adapter tensor: {k}")
    return {"passed": True, "tensor_count": len(saved), "parameters": sum(v.numel() for v in saved.values())}


def check_dataset(dataset, records):
    require(len(dataset) == len(records), "Training preprocessor dropped or duplicated rows")
    for i, r in enumerate(records):
        row = dataset[i]
        ids, labels = list(row["input_ids"]), list(row["labels"])
        require(len(ids) == len(labels) == r["total_tokens"] <= 4096
                and digest(ids) == r["input_ids_sha256"] and digest(labels) == r["labels_sha256"],
                f"Actual training tokens/labels differ from audited row {r['id']}")
    return {"passed": True, "rows": len(records), "assistant_turns": sum(r["assistant_turns"] for r in records),
            "ordered_audit_sha256": digest(records)}
