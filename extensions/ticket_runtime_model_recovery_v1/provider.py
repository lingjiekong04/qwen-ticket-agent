from __future__ import annotations

from . import support as s


class HFProvider:
    def __init__(self, model=None, tokenizer=None, template=None):
        self.model = model
        self.tokenizer = model.engine.tokenizer if model else tokenizer
        self.template = model.engine.template if model else template

    def prompt_info(self, payload: dict) -> dict:
        ids, _ = self.template.encode_oneturn(
            self.tokenizer, payload["messages"] + [{"role": "assistant", "content": ""}],
            payload["system"], payload["tools"])
        s.require(bool(ids), "Empty encoded model input")
        return {"tokens": len(ids), "token_ids_sha256": s.digest(list(ids)), "kind": "actual_qwen_token_ids"}

    def generate(self, payload: dict) -> dict:
        output = self.model.chat(messages=payload["messages"], system=payload["system"],
                                 tools=payload["tools"], do_sample=False, max_new_tokens=512,
                                 repetition_penalty=1.0, num_return_sequences=1)
        s.require(len(output) == 1 and isinstance(output[0].response_text, str), "Wrong HF response shape")
        result = output[0]
        s.require(type(result.response_length) is int and 0 <= result.response_length <= 512,
                  "Wrong generation length")
        return {"text": result.response_text, "finish_reason": result.finish_reason,
                "prompt_tokens": result.prompt_length, "response_tokens": result.response_length}


def token_provider() -> HFProvider:
    from transformers import AutoTokenizer
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments
    tokenizer = AutoTokenizer.from_pretrained(str(s.MODEL), local_files_only=True, trust_remote_code=False,
                                               use_fast=True, padding_side="right", split_special_tokens=False)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="qwen"))
    return HFProvider(tokenizer=tokenizer, template=template)


def check_loaded(model, expected_uuid: str) -> None:
    import torch
    s.require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(),
              "Require one visible BF16 GPU")
    actual_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    s.require(actual_uuid is not None and str(actual_uuid).removeprefix("GPU-").lower()
              == expected_uuid.removeprefix("GPU-").lower(), "Logical GPU differs from physical GPU0")
    s.require(not torch.distributed.is_initialized(), "Distributed inference unexpectedly active")
    actual = model.engine.model
    s.require(not getattr(actual, "is_loaded_in_4bit", False) and not getattr(actual, "is_loaded_in_8bit", False),
              "Unexpected quantized inference")
    s.require(getattr(actual, "_tp_size", 1) in (None, 1), "Tensor parallel unexpectedly active")
    s.require(all(parameter.device.type == "cuda" and parameter.device.index == 0
                  and not parameter.requires_grad for parameter in actual.parameters()),
              "Model offloaded, on wrong device, or trainable during inference")

