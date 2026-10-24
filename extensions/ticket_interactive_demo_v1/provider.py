"""Pinned local LLaMA-Factory provider; heavy imports only on explicit startup."""
from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys

from . import core

CONFIG = core.ROOT / "configs/ticket_interactive_demo_v1.json"


def verify_sources() -> dict:
    binding = core.load(core.HERE / "bindings.json")
    for name, expected in binding["files_sha256"].items():
        path = core.ROOT / name
        core.require(path.is_file() and core.sha(path) == expected, "Source changed or missing: " + name)
    return {"passed": True, "files_verified": len(binding["files_sha256"]),
            "runtime": "trusted_completion_v1.7", "model_loaded": False}


def verify_server() -> dict:
    config = core.load(CONFIG)
    core.require(sys.version_info[:2] == (3, 11), "Use the pinned Python 3.11 environment")
    versions = {key: importlib.metadata.version(key) for key in config["stack"]}
    core.require(versions == config["stack"], "Inference package versions differ from frozen stack")
    from pathlib import Path
    for directory, values in ((config["model_path"], config["base_files_sha256"]),
                              (config["model_path"], config["tokenizer_files_sha256"]),
                              (config["adapter_path"], config["adapter_files_sha256"])):
        for name, expected in values.items():
            core.require(core.sha(Path(directory) / name) == expected, "Weight/tokenizer changed: " + name)
    return {"passed": True, "stack": versions, "weights_verified": True, "model_loaded": False}


def load_local_provider():
    config = core.load(CONFIG)
    verify_sources()
    verify_server()
    core.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Select physical GPU0 only")
    for key in list(os.environ):
        if key in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                   "GROUP_RANK", "ROLE_RANK", "NODE_RANK"} or key.startswith(("SLURM_", "ACCELERATE_", "PMI", "OMPI_", "TORCHELASTIC_")):
            os.environ.pop(key)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      FORCE_TORCHRUN="0", TOKENIZERS_PARALLELISM="false")
    def query(args):
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True, timeout=20).stdout
    fields = query(["--id=0", "--query-gpu=uuid,name,memory.free", "--format=csv,noheader,nounits"]).strip().split(",")
    gpu_uuid, name, free = [x.strip() for x in fields]
    core.require("A100" in name and float(free) >= 30000, "GPU0 must be an idle A100 with at least 30GB free")
    processes = query(["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
    core.require(not any(line.split(",")[0].strip() == gpu_uuid for line in processes.splitlines()), "GPU0 is occupied")
    from llamafactory.chat import ChatModel
    from transformers import set_seed
    import torch
    set_seed(2026102208)
    model = ChatModel({
        "model_name_or_path": config["model_path"], "adapter_name_or_path": config["adapter_path"],
        "create_new_adapter": False, "template": "qwen", "infer_backend": "huggingface",
        "finetuning_type": "lora", "trust_remote_code": False, "infer_dtype": "bfloat16",
        "flash_attn": "sdpa", "do_sample": False, "max_new_tokens": 512, "repetition_penalty": 1.0,
    })
    core.require(torch.cuda.device_count() == 1 and not torch.distributed.is_initialized(), "Unexpected distributed GPU state")
    actual_uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    core.require(actual_uuid.removeprefix("GPU-").lower() == gpu_uuid.removeprefix("GPU-").lower(), "GPU UUID mismatch")
    actual = model.engine.model
    core.require(not getattr(actual, "is_loaded_in_4bit", False) and
                 not getattr(actual, "is_loaded_in_8bit", False), "Unexpected quantized model")
    actual.eval()
    for parameter in actual.parameters():
        parameter.requires_grad_(False)
    core.require(all(p.device.type == "cuda" and p.device.index == 0 for p in actual.parameters()), "Model not wholly on GPU0")
    return LocalQwenProvider(model, torch)


class LocalQwenProvider:
    def __init__(self, model, torch_module):
        self.model, self.torch = model, torch_module

    def prompt_info(self, payload):
        ids, _ = self.model.engine.template.encode_oneturn(
            self.model.engine.tokenizer, payload["messages"] + [{"role": "assistant", "content": ""}],
            payload["system"], payload["tools"])
        core.require(bool(ids) and len(ids) + 512 <= 8192, "演示上下文预算超限")
        return {"kind": "actual_qwen_token_ids", "tokens": len(ids), "sha256": core.wire.digest(list(ids))}

    def generate(self, payload):
        with self.torch.inference_mode():
            output = self.model.chat(messages=payload["messages"], system=payload["system"], tools=payload["tools"],
                                     do_sample=False, max_new_tokens=512, repetition_penalty=1.0, num_return_sequences=1)
        core.require(len(output) == 1, "Local Qwen returned unexpected count")
        item = output[0]
        core.require(isinstance(item.response_text, str) and type(item.response_length) is int and
                     0 <= item.response_length <= 512, "Unexpected local Qwen output shape")
        return {"text": item.response_text, "finish_reason": item.finish_reason,
                "prompt_tokens": item.prompt_length, "response_tokens": item.response_length}
