"""Separate 144-episode inference diagnostic; never imports a training launcher."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from . import core as c
s = c.s


def prepare():
    c.delivery_gate()
    rows = c.build_tasks()
    population = c.validate_tasks(rows, s.frozen_descriptor()["tasks"])
    s.fresh(c.OUTPUT)
    c.OUTPUT.mkdir(parents=True)
    s.write(c.OUTPUT / "tasks.json", rows)
    s.write(c.OUTPUT / "manifest.json", {"scope": "post_DEV_diagnostic_not_training_or_final_test", "population": population,
        "tasks_sha256": s.sha(c.OUTPUT / "tasks.json"), "sources": c.sources(), "policy": c.POLICY,
        "frozen_dev_sha256": s.DEV_SHA, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    return population


def prepared():
    c.delivery_gate()
    manifest = s.read(c.OUTPUT / "manifest.json")
    rows = s.read(c.OUTPUT / "tasks.json")
    c.validate_tasks(rows, s.frozen_descriptor()["tasks"])
    s.require(manifest["tasks_sha256"] == s.sha(c.OUTPUT / "tasks.json") and manifest["sources"] == c.sources()
              and manifest["policy"] == c.POLICY and manifest["frozen_dev_sha256"] == s.DEV_SHA, "Diagnostic preparation changed")
    return rows, manifest


def artifacts():
    # Reuses immutable acceptance checks only, never calls train/execute.
    from extensions.ticket_training_v1_1.train import check_completed
    check_completed("train")
    checked = s.artifact_gate()
    adapter = s.RUN / "train/adapter"
    s.require(s.sha(adapter / "adapter_model.safetensors") == c.NEW_SHA, "Not the reviewed final v3 adapter")
    s.check_adapter(s.read(adapter / "adapter_config.json"))
    return {**checked, "new_adapter_sha256": c.NEW_SHA, "new_adapter_config_sha256": s.sha(adapter / "adapter_config.json"),
            "training_run_sha256": s.sha(s.RUN / "train/run.json")}


def model_args(variant):
    s.require(variant in c.VARIANTS, "Unknown adapter")
    return {"model_name_or_path": s.MODEL, "adapter_name_or_path": s.OLD if variant == "hermes-r8" else str((s.RUN / "train/adapter").resolve()),
            "template": "qwen", "infer_backend": "huggingface", "finetuning_type": "lora", "trust_remote_code": False,
            "infer_dtype": "bfloat16", "flash_attn": "sdpa", "do_sample": False, "max_new_tokens": 512, "repetition_penalty": 1.0}


def evaluate(variant):
    tasks, manifest = prepared()
    output = c.OUTPUT / variant
    s.fresh(output)
    # Runtime admission happens before reading model weights or loading CUDA.
    versions, gpu = s.runtime_gate()
    evidence = artifacts()
    from transformers import set_seed
    from llamafactory.chat import ChatModel
    set_seed(42)
    model = ChatModel(model_args(variant))
    provider = s.legacy().HFProvider(model)
    runner = c.isolated_agent()
    initial = {t["id"]: provider.prompt_info(runner.input_payload([{"role": "user", "content": t["request"]}], c.system_prompt(t))) for t in tasks}
    s.require(all(i["tokens"] + 512 <= 8192 for i in initial.values()), "Initial context exceeds fixed budget; do not drop cases")
    output.mkdir()
    (output / "progress").mkdir()
    run = {"status": "started", "variant": variant, "scope": "post_DEV_diagnostic", "provider_kind": "huggingface",
           "started_at_utc": datetime.now(timezone.utc).isoformat(), "model_args": model_args(variant), "policy": c.POLICY,
           "versions": versions, "allocated_gpu": gpu, "artifact_evidence": evidence,
           "manifest_sha256": s.sha(c.OUTPUT / "manifest.json"), "sources": c.sources(),
           "initial_prompt_audit": initial, "case_ids": [t["id"] for t in tasks], "human_review": "pending"}
    s.write(output / "run.json", run)
    try:
        records = []
        with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
            for index, task in enumerate(tasks):
                record = runner.run_episode(task, provider, progress_path=output / "progress" / f"{task['id']}.json")
                runner.replay_record(task, record)
                s.require(record["rounds"][0]["prompt_info"] == initial[task["id"]], "Initial tokenizer input changed")
                stream.write(c.canonical(record) + "\n")
                stream.flush()
                records.append(record)
                print(f"Diagnostic {variant} {index+1}/72: {task['id']} -> {record['termination']}", flush=True)
        s.write(output / "metrics.json", c.summary(records, tasks))
        s.write(output / "answer-review.json", [{"id": r["id"], "request": t["request"], "profile": t["prompt_profile"],
            "answer": r["final_text"], "call_count": len(r["events"]), "human_review": "pending",
            "semantic_review": "pending", "ai_review": "not_started"} for t, r in zip(tasks, records)])
        s.require(prepared()[1] == manifest and artifacts() == evidence, "Sources or model artifacts changed during inference")
        run.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                   output_sha256={name: s.sha(output / name) for name in ("episodes.jsonl", "metrics.json", "answer-review.json")})
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        s.update(output / "run.json", run)
        raise
    s.update(output / "run.json", run)


def replay_variant(variant, tasks, evidence):
    directory = c.OUTPUT / variant
    run = s.read(directory / "run.json")
    s.require(run["status"] == "complete" and run["variant"] == variant and run["scope"] == "post_DEV_diagnostic"
              and run["provider_kind"] == "huggingface" and run["versions"] == s.STACK
              and run["artifact_evidence"] == evidence and run["sources"] == c.sources()
              and run["policy"] == c.POLICY and run["model_args"] == model_args(variant)
              and run["manifest_sha256"] == s.sha(c.OUTPUT / "manifest.json")
              and run["case_ids"] == [t["id"] for t in tasks], "Diagnostic run provenance differs")
    s.require(run["allocated_gpu"]["selection"] == "1" and run["allocated_gpu"]["uuid"].startswith("GPU-"), "Wrong GPU")
    s.require(set(run["output_sha256"]) == {"episodes.jsonl", "metrics.json", "answer-review.json"}, "Missing output hashes")
    for name, expected in run["output_sha256"].items():
        s.require(s.sha(directory / name) == expected, "Diagnostic output changed")
    rows = [c.strict_json(line) for line in (directory / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    s.require([r["id"] for r in rows] == [t["id"] for t in tasks], "Case filtering/reordering detected")
    runner = c.isolated_agent()
    replayed = [runner.replay_record(t, r) for t, r in zip(tasks, rows)]
    # Re-encode every recorded history with the actual pinned tokenizer in the
    # comparison process, without loading a model or regenerating outputs.
    from llamafactory.hparams import ModelArguments, DataArguments
    from llamafactory.model import load_tokenizer
    from llamafactory.data import get_template_and_fix_tokenizer
    tokenizer = load_tokenizer(ModelArguments(model_name_or_path=s.MODEL, trust_remote_code=False))["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="qwen"))
    provider = s.legacy().HFProvider(tokenizer=tokenizer, template=template)
    for row in rows:
        s.require(run["initial_prompt_audit"][row["id"]] == row["rounds"][0]["prompt_info"], "Initial audit differs")
        for step in row["rounds"]:
            s.require(provider.prompt_info(step["input"]) == step["prompt_info"], "Actual tokenizer/history audit differs")
    metrics = c.summary(replayed, tasks)
    s.require(metrics == s.read(directory / "metrics.json"), "Recorded metrics differ from independent replay")
    return run, metrics, rows


def compare():
    tasks, _ = prepared()
    target = c.OUTPUT / "comparison.json"
    s.fresh(target)
    evidence = artifacts()
    results = {v: replay_variant(v, tasks, evidence) for v in c.VARIANTS}
    old, new = (results[v][0] for v in c.VARIANTS)
    s.require(old["allocated_gpu"]["uuid"] == new["allocated_gpu"]["uuid"]
              and datetime.fromisoformat(old["finished_at_utc"]) <= datetime.fromisoformat(new["started_at_utc"]), "Require sequential matched-GPU runs")
    indexed = {v: {r["id"]: r for r in results[v][2]} for v in c.VARIANTS}
    changes = [{"id": t["id"], "profile": t["prompt_profile"],
                "old_safe": indexed["hermes-r8"][t["id"]]["scores"]["safe_behavior"],
                "new_safe": indexed["ticket-sft-v3"][t["id"]]["scores"]["safe_behavior"]} for t in tasks]
    s.write(target, {"scope": "Post-DEV prompt sensitivity only; never replaces original DEV scores or proves a causal training fix.",
        "rows_per_adapter": 72, "metrics": {v: results[v][1] for v in c.VARIANTS}, "changes": changes,
        "run_sha256": {v: s.sha(c.OUTPUT / v / "run.json") for v in c.VARIANTS},
        "manifest_sha256": s.sha(c.OUTPUT / "manifest.json"), "replay_passed": True, "actual_tokenizer_reaudit_passed": True,
        "answer_semantics": "pending", "human_review": "pending", "training_started": False})
    print(json.dumps(s.read(target), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "hermes-r8", "ticket-sft-v3", "compare"), required=True)
    args = parser.parse_args()
    if args.stage == "prepare": print(prepare())
    elif args.stage == "compare": compare()
    else: evaluate(args.stage)
