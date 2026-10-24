"""Real pinned SFT with checks inside the actual Trainer before any update."""
import argparse
import gc
from pathlib import Path
import sys

from . import support as s


def build_callback(stage_dir, records, steps):
    from transformers import TrainerCallback
    from safetensors.torch import load_file
    import torch

    class EvidenceCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            s.require(state.global_step == 0 and state.max_steps == steps and args.resume_from_checkpoint is None,
                      "Optimizer state/resume or planned steps differ")
            optimizer = kwargs["optimizer"]
            s.require(not optimizer.state, "Optimizer inherited prior state")
            model = kwargs["model"]
            trainable = {k: p for k, p in model.named_parameters() if p.requires_grad}
            s.require(all(".lora_A." in k or ".lora_B." in k for k in trainable), "Unexpected trainable base parameters")
            result = s.compare_tensor_sets(load_file(str(Path(s.OLD) / "adapter_model.safetensors"), device="cpu"), trainable)
            result.update(initializer_sha256=s.OLD_SHA, optimizer_fresh=True, global_step=0)
            s.write(stage_dir / "initialization.json", result)
            loader = kwargs["train_dataloader"]
            actual = s.check_dataset(loader.dataset, records)
            s.write(stage_dir / "training-input-audit.json", actual)
            s.require(len(loader) == len(records), "Unexpected batch count/drop_last")
            print(f"Actual Trainer inputs checked: {len(records)} rows; all assistant labels retained.", flush=True)

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            for name, param in kwargs["model"].named_parameters():
                if param.requires_grad and param.grad is not None:
                    s.require(torch.isfinite(param.grad).all().item(), f"Nonfinite gradient: {name}")

        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ("loss", "train_loss", "grad_norm", "learning_rate"):
                if logs and key in logs:
                    s.require(s.finite(logs[key]), f"Nonfinite {key}; retain failed output")

    return EvidenceCallback()


def prepare_inputs(stage, directory):
    episodes = s.read(s.CANDIDATE / "episodes.json")
    audit = s.read(s.RELEASE / "token-audit.json")
    s.require([e["id"] for e in episodes] == [r["id"] for r in audit["records"]], "Training/audit population differs")
    if stage == "train":
        return audit["records"]
    indices, longest = s.select_preflight(episodes, audit["records"])
    data = directory / "data"
    s.fresh(data)
    data.mkdir()
    s.write(data / "train.json", [episodes[i]["row"] for i in indices])
    registration = s.read(s.RELEASE / "dataset_info.json")["ticket_sft_v3_train"]
    s.write(data / "dataset_info.json", {"ticket_sft_v3_preflight": registration})
    s.write(directory / "sample.json", {"seed": 42, "pair_count": 40, "rows": 80,
            "ids": [episodes[i]["id"] for i in indices], "longest_pair_by_category": longest,
            "release_manifest_sha256": s.RELEASE_SHA, "train_sha256": s.sha(data / "train.json"),
            "scope": "Subset of training only; no DEV and never the formal initializer"})
    return [audit["records"][i] for i in indices]


def verify_result(directory, stage):
    adapter = directory / "adapter"
    s.check_state(s.read(adapter / "trainer_state.json"), s.read(adapter / "all_results.json"),
                  10 if stage == "preflight" else 750, 1.0 if stage == "preflight" else 2.0)
    s.check_adapter(s.read(adapter / "adapter_config.json"))
    weights = adapter / "adapter_model.safetensors"
    s.require(weights.is_file() and weights.stat().st_size > 0 and s.sha(weights) != s.OLD_SHA,
              "Adapter missing/empty/unchanged")
    expected_rows = 80 if stage == "preflight" else 3000
    input_audit, initial = s.read(directory / "training-input-audit.json"), s.read(directory / "initialization.json")
    s.require(input_audit["passed"] is True and input_audit["rows"] == expected_rows
              and initial["passed"] is True and initial["optimizer_fresh"] is True
              and initial["initializer_sha256"] == s.OLD_SHA, "Trainer input/initialization evidence missing")
    s.require(s.sha(Path(s.OLD) / "adapter_model.safetensors") == s.OLD_SHA, "Old adapter modified")
    return {"passed": True, "rows": expected_rows, "global_step": 10 if stage == "preflight" else 750,
            "epoch": 1 if stage == "preflight" else 2, "train_loss": s.read(adapter / "all_results.json")["train_loss"],
            "adapter_sha256": s.sha(weights), "adapter_config_sha256": s.sha(adapter / "adapter_config.json"),
            "adapter_bytes": weights.stat().st_size, "initializer_unchanged": True}


def reload_check(adapter, config, first_ids):
    """Load unmerged LoRA for exact tensor equality, then run a finite inference forward."""
    from llamafactory.hparams import ModelArguments, FinetuningArguments
    from llamafactory.model import load_model, load_tokenizer
    from safetensors.torch import load_file
    import torch
    args = ModelArguments(model_name_or_path=s.MODEL, adapter_name_or_path=[str(adapter.resolve())],
                          trust_remote_code=False, flash_attn="sdpa")
    args.compute_dtype = torch.bfloat16
    finetune = FinetuningArguments(finetuning_type="lora", create_new_adapter=False)
    tokenizer = load_tokenizer(args)["tokenizer"]
    # is_trainable preserves the unmerged adapter. There is no Trainer or optimizer here.
    model = load_model(tokenizer, args, finetune, is_trainable=True)
    model.to(torch.device("cuda:0"))  # logical device 0 is the explicitly selected physical GPU 1
    result = s.compare_tensor_sets(load_file(str(adapter / "adapter_model.safetensors"), device="cpu"),
                                  {k: p for k, p in model.named_parameters() if p.requires_grad})
    model.eval()
    ids = torch.tensor([first_ids[:32]], dtype=torch.long, device=next(model.parameters()).device)
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        s.require(torch.isfinite(output.logits).all().item(), "Reloaded adapter inference logits nonfinite")
    result.update(finite_forward=True, adapter_sha256=s.sha(adapter / "adapter_model.safetensors"))
    del model, output, ids
    gc.collect()
    torch.cuda.empty_cache()
    return result


def check_completed(stage):
    directory = s.RUN / stage
    run = s.read(directory / "run.json")
    s.require(run["status"] == "complete" and run["stage"] == stage and run["sources"] == s.local_sources()
              and run["release_manifest_sha256"] == s.RELEASE_SHA and run["initializer_sha256"] == s.OLD_SHA,
              "Prerequisite training run is incomplete or changed")
    for name, expected in run["outputs_sha256"].items():
        s.require(s.sha(directory / name) == expected, f"Training evidence changed: {name}")
    current = verify_result(directory, stage)
    s.require(s.read(directory / "verification.json") == {**current, "reload": s.read(directory / "reload.json")},
              "Verification differs from saved training result")
    s.require(s.read(directory / "reload.json")["finite_forward"] is True, "Missing actual reload evidence")
    return run


def execute(stage):
    import yaml
    config_path = s.ROOT / f"configs/ticket_sft_v3_{stage}_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    s.check_config(config, stage)
    directory = s.RUN / stage
    s.fresh(directory)
    if stage == "train":
        check_completed("preflight")
    release = s.release_gate()
    versions, gpu = s.runtime_gate()
    artifacts = s.artifact_gate()
    # DEV is checked for immutability/provenance but never fed to the trainer.
    _, _, dev_meta = s.dev_gate()
    directory.mkdir(parents=True)
    run = {"status": "started", "stage": stage, "started_at_utc": s.now(), "versions": versions,
           "allocated_gpu": gpu, "config": config, "config_sha256": s.sha(config_path),
           "sources": s.local_sources(), "release_manifest_sha256": release["release_manifest_sha256"],
           **artifacts, **dev_meta, "initialization": "Old Hermes r8 weights; new optimizer/scheduler; never preflight weights",
           "selection": "Fixed two epochs and final adapter; no DEV loss/checkpoint selection"}
    s.write(directory / "run.json", run)
    try:
        records = prepare_inputs(stage, directory)
        from transformers import set_seed
        from llamafactory.hparams import get_train_args
        from llamafactory.train.sft.workflow import run_sft
        set_seed(42)
        args = get_train_args(config)
        s.require(args[2].resume_from_checkpoint is None, "No checkpoint resume permitted")
        # CPU audit against the original token hashes is repeated inside the actual Trainer callback.
        run_sft(*args, callbacks=[build_callback(directory, records, 10 if stage == "preflight" else 750)])
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        result = verify_result(directory, stage)
        # Encode one source row for the reload forward without using DEV or filtering training rows.
        from llamafactory.data import get_template_and_fix_tokenizer
        from llamafactory.hparams import DataArguments, ModelArguments
        from llamafactory.model import load_tokenizer
        tokenizer = load_tokenizer(ModelArguments(model_name_or_path=s.MODEL))["tokenizer"]
        template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="qwen"))
        row = s.read(Path(config["dataset_dir"]) / "train.json")[0]
        roles = {"human": "user", "gpt": "assistant", "function_call": "function", "observation": "observation"}
        pairs = template.encode_multiturn(tokenizer, [{"role": roles[m["from"]], "content": m["value"]}
                                                    for m in row["conversations"]], row["system"], row["tools"])
        reload = reload_check(directory / "adapter", config, pairs[0][0])
        s.write(directory / "reload.json", reload)
        s.write(directory / "verification.json", {**result, "reload": reload})
        s.require(s.local_sources() == run["sources"] and s.sha(s.RELEASE / "manifest.json") == s.RELEASE_SHA,
                  "Sources or released data changed during training")
        s.require(s.artifact_gate() == artifacts and s.dev_gate()[2] == dev_meta, "Immutable inputs changed during training")
        names = ["adapter/adapter_config.json", "adapter/adapter_model.safetensors", "adapter/trainer_state.json",
                 "adapter/all_results.json", "initialization.json", "training-input-audit.json", "reload.json", "verification.json"]
        if stage == "preflight":
            names += ["sample.json", "data/train.json", "data/dataset_info.json"]
        run.update(status="complete", finished_at_utc=s.now(), outputs_sha256={p: s.sha(directory / p) for p in names})
        s.update(directory / "run.json", run)
        print(s.read(directory / "verification.json"), flush=True)
    except BaseException as exc:
        run.update(status="failed", finished_at_utc=s.now(), error=f"{type(exc).__name__}: {exc}")
        s.update(directory / "run.json", run)
        raise


if __name__ == "__main__":
    import signal
    def interrupted(signum, frame):
        raise SystemExit(f"Interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "train"), required=True)
    import os
    os.chdir(s.ROOT)
    execute(parser.parse_args().stage)
