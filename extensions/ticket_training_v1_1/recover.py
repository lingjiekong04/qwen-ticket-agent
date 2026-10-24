"""Import completed v1 preflight evidence, rerun reload ONLY. Never train or rewrite v1."""
from pathlib import Path
import shutil

from . import support as s
from . import train
from extensions.ticket_training_v1 import support as old
from extensions.ticket_training_v1 import train as old_train

ERROR = "AttributeError: 'list' object has no attribute 'split'"
COPY_FILES = ("adapter/adapter_config.json", "adapter/adapter_model.safetensors", "adapter/trainer_state.json",
              "adapter/all_results.json", "initialization.json", "training-input-audit.json", "sample.json",
              "data/train.json", "data/dataset_info.json")


def check_failed_run(run, sources, config_sha):
    s.require(run["status"] == "failed" and run["stage"] == "preflight" and run.get("error") == ERROR,
              "Only the identified v1 post-training list/split reload failure can be recovered")
    s.require(run["sources"] == sources and run["config"] == old.config_for("preflight")
              and run["config_sha256"] == config_sha and run["release_manifest_sha256"] == s.RELEASE_SHA
              and run["initializer_sha256"] == s.OLD_SHA, "Failed-run source/configuration differs")


def check_bridge(directory, run):
    source = old.RUN / "preflight"
    original = s.read(directory / "original-run.json")
    check_failed_run(original, old.local_sources(), old.sha(old.ROOT / "configs/ticket_sft_v3_preflight_v1.yaml"))
    expected = {n: s.sha(source / n) for n in COPY_FILES + ("run.json",)}
    s.require(run.get("training_executed_in_recovery") is False
              and run["original_outputs_sha256"] == expected and run["original_run_sha256"] == expected["run.json"]
              and s.sha(directory / "original-run.json") == expected["run.json"]
              and run["original_training_sources"] == original["sources"] and run["config"] == original["config"]
              and run["config_sha256"] == original["config_sha256"], "Recovery bridge/source failure changed")
    for name in COPY_FILES:
        s.require(s.sha(directory / name) == expected[name], "Imported trained evidence differs from original")
    verify_preflight_inputs(directory)


def verify_preflight_inputs(directory):
    episodes = s.read(s.CANDIDATE / "episodes.json")
    audit = s.read(s.RELEASE / "token-audit.json")["records"]
    indices, longest = old.select_preflight(episodes, audit)
    records = [audit[i] for i in indices]
    expected_sample = {"seed": 42, "pair_count": 40, "rows": 80, "ids": [episodes[i]["id"] for i in indices],
                       "longest_pair_by_category": longest, "release_manifest_sha256": s.RELEASE_SHA,
                       "train_sha256": s.sha(directory / "data/train.json"),
                       "scope": "Subset of training only; no DEV and never the formal initializer"}
    s.require(s.read(directory / "sample.json") == expected_sample
              and s.read(directory / "data/train.json") == [episodes[i]["row"] for i in indices],
              "Original preflight sample/rows differ from the fixed 80 training rows")
    registration = s.read(s.RELEASE / "dataset_info.json")["ticket_sft_v3_train"]
    s.require(s.read(directory / "data/dataset_info.json") == {"ticket_sft_v3_preflight": registration},
              "Original preflight registration differs")
    expected_audit = {"passed": True, "rows": 80, "assistant_turns": sum(r["assistant_turns"] for r in records),
                      "ordered_audit_sha256": s.digest(records)}
    s.require(s.read(directory / "training-input-audit.json") == expected_audit,
              "Actual v1 Trainer token/label audit differs from the fixed source audit")
    initial = s.read(directory / "initialization.json")
    s.require(initial.get("global_step") == 0 and initial.get("optimizer_fresh") is True
              and initial.get("passed") is True and initial.get("initializer_sha256") == s.OLD_SHA
              and initial.get("tensor_count", 0) > 0 and initial.get("parameters", 0) > 0,
              "Original preflight initial tensor/fresh optimizer evidence incomplete")
    return records


def execute():
    source = old.RUN / "preflight"
    directory = s.RUN / "preflight"
    s.fresh(directory)
    old.delivery_gate()
    original = old.read(source / "run.json")
    config_path = old.ROOT / "configs/ticket_sft_v3_preflight_v1.yaml"
    check_failed_run(original, old.local_sources(), old.sha(config_path))
    # This checks 10 completed steps, finite metrics, 80 actual inputs and unchanged old initializer.
    old_result = old_train.verify_result(source, "preflight")
    verify_preflight_inputs(source)
    original_hashes = {name: s.sha(source / name) for name in COPY_FILES + ("run.json",)}
    s.release_gate()
    versions, gpu = s.runtime_gate()
    artifacts = s.artifact_gate()
    _, _, dev_meta = s.dev_gate()
    for key, value in {**artifacts, **dev_meta}.items():
        s.require(original[key] == value, f"Original training input provenance changed: {key}")
    directory.mkdir(parents=True)
    run = {"status": "started", "stage": "preflight", "started_at_utc": s.now(), "versions": versions,
           "allocated_gpu": gpu, "sources": s.local_sources(), **artifacts, **dev_meta,
           "release_manifest_sha256": s.RELEASE_SHA, "config": original["config"],
           "config_sha256": original["config_sha256"], "original_run_sha256": original_hashes["run.json"],
           "original_outputs_sha256": original_hashes, "original_training_sources": original["sources"],
           "execution": "Imported v1's already-completed 10 steps; v1.1 performs reload verification only",
           "training_executed_in_recovery": False, "scope": "v1 failure is preserved; no optimizer resume or adapter update"}
    s.write(directory / "run.json", run)
    try:
        for name in COPY_FILES:
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            s.fresh(target)
            shutil.copyfile(source / name, target)
            s.require(s.sha(target) == original_hashes[name], "Copied evidence differs")
        shutil.copyfile(source / "run.json", directory / "original-run.json")
        result = train.verify_result(directory, "preflight")
        s.require(result == old_result, "Copied trained artifacts differ from original")
        from llamafactory.hparams import ModelArguments, DataArguments
        from llamafactory.model import load_tokenizer
        from llamafactory.data import get_template_and_fix_tokenizer
        tokenizer = load_tokenizer(ModelArguments(model_name_or_path=s.MODEL, trust_remote_code=False))["tokenizer"]
        template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="qwen"))
        row = s.read(directory / "data/train.json")[0]
        roles = {"human": "user", "gpt": "assistant", "function_call": "function", "observation": "observation"}
        pairs = template.encode_multiturn(tokenizer, [{"role": roles[m["from"]], "content": m["value"]}
                                                    for m in row["conversations"]], row["system"], row["tools"])
        reload = train.reload_check(directory / "adapter", original["config"], pairs[0][0])
        s.write(directory / "reload.json", reload)
        s.write(directory / "verification.json", {**result, "reload": reload})
        s.require(original_hashes == {name: s.sha(source / name) for name in original_hashes},
                  "Original failure evidence changed during recovery")
        s.require(s.local_sources() == run["sources"] and s.artifact_gate() == artifacts
                  and s.dev_gate()[2] == dev_meta, "Recovery inputs changed")
        output_names = COPY_FILES + ("original-run.json", "reload.json", "verification.json")
        run.update(status="complete", finished_at_utc=s.now(), outputs_sha256={p: s.sha(directory / p) for p in output_names})
        check_bridge(directory, run)
        s.update(directory / "run.json", run)
        print({"recovered": True, "training_executed_in_recovery": False,
               "original_failure_preserved": True, "preflight_steps": 10, "reload": reload}, flush=True)
    except BaseException as exc:
        run.update(status="failed", finished_at_utc=s.now(), error=f"{type(exc).__name__}: {exc}")
        s.update(directory / "run.json", run)
        raise


if __name__ == "__main__":
    import os
    import signal
    def interrupted(signum, frame): raise SystemExit(f"Interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    os.chdir(s.ROOT)
    execute()
