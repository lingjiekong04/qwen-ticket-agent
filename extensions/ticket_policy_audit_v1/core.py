"""CPU-only, append-only permission and supervision evidence audit."""
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from extensions.ticket_policy_diagnostic_v1_1 import core as diagnostic
s = diagnostic.s
ROOT = s.ROOT
HERE = Path(__file__).resolve().parent
OUT = ROOT / "reports/ticket-policy-audit-v1"
TRAIN_RUN_SHA = "f267c8e3dab8436ff69c5cc102dec484aa7ad9aa8a5713c5ea28db6ead33e3a7"
UPLOADS = {
    "comparison.json": "10678e6e6c2e0387d0c835b02541967e0b9212068939d9b619b8d83df4f3e81e",
    "episodes-hermes-r8.jsonl": "e836c206158723b1b9a40faa369cc95d74eb02b984fcbca5bce3823cde439b51",
    "episodes-ticket-sft-v3.jsonl": "d52561fc8df96ab4fba14bce741b154fcc93174c70dbac62d82a081e7f355ea3",
    "run-hermes-r8.json": "f7714d560646aee3a2a714f657b0f7395c7b2fa144434689bf8f0dd71ea6112e",
    "run-ticket-sft-v3.json": "9617c1883f46a1b345d5059b8bbef6908d602d7aae81f5635589b8bcaf759b7e",
}
ROLES = {"human": "user", "gpt": "assistant", "function_call": "function", "observation": "observation"}
SYSTEM = {
    "zh": "你是支持助手。仅用公开工具和已观察返回。缺参先询问；无写权限直接拒绝。独立查询可批量，依赖调用等待结果。冲突后重新读取，最多重试两次。无结果或其他错误及时停止。如实报告实际状态，不编造分配、来源或后续承诺。",
    "en": "You are a support assistant. Use only public tools and observed results. Ask for missing fields; refuse writes when unauthorized. Batch independent reads only; wait for dependencies. Re-read after conflicts, retry at most twice. Stop on empty results or other errors. Report actual states without invented assignments, sources or future promises.",
}
MARKER = "Trusted execution policy: "
TOOLS = diagnostic.original_agent.TOOLS
# Framework data utilities may import model class definitions. That alone is
# not weight loading; prohibit executable chat/training entrypoints instead.
BLOCKED_IMPORTS = ("llamafactory.chat", "llamafactory.train")


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sources():
    paths = sorted(HERE.rglob("*.py")) + [ROOT / "scripts/run_ticket_policy_audit_v1.py"]
    return {p.relative_to(ROOT).as_posix(): s.sha(p) for p in paths}


def cpu_environment(parent):
    clean, removed = diagnostic.child_environment(parent)
    clean.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                 HF_DATASETS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    return clean, removed


def cpu_gate(environment):
    s.require(environment.get("CUDA_VISIBLE_DEVICES") == "" and not diagnostic.distributed_keys(environment),
              "CPU audit requires GPUs hidden and distributed variables removed by launcher")
    s.require(all(environment.get(k) == "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")),
              "Require offline model/data loading")


class CPUOnlyImports:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == x or fullname.startswith(x + ".") for x in BLOCKED_IMPORTS):
            raise ImportError("Audit forbids model/training import: " + fullname)
        return None


def private_stage2(name):
    # These predecessor modules use generic core/pipeline imports. Run this only
    # inside a fresh audit child; do not modify any predecessor file.
    sys.path.insert(0, str(ROOT / "extensions/ticket_sft_v3_stage2"))
    spec = importlib.util.spec_from_file_location("ticket_policy_audit_" + name,
        ROOT / f"extensions/ticket_sft_v3_stage2/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_gate():
    diagnostic.delivery_gate()
    s.require(s.sha(s.RELEASE / "manifest.json") == s.RELEASE_SHA, "Wrong release manifest")
    result = subprocess.run([sys.executable, "-B", "scripts/verify_ticket_sft_stage2_release.py"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=600)
    s.require(result.returncode == 0, "Release verification failed: " + result.stderr)
    release = json.loads(result.stdout)
    s.require(release["passed"] and release["rows"] == 3000, "Incomplete release")
    return release


def permission(row, expected, locale):
    s.require(set(row) == {"system", "tools", "conversations"}, "Hidden metadata in training row")
    text = row["system"]
    s.require(text.count(MARKER) == 1, "Missing/duplicate trusted permission")
    prefix, raw = text.split("\n" + MARKER)
    policy = json.loads(raw)
    s.require(type(policy.get("allow_write")) is bool and type(policy.get("max_calls")) is int
              and policy == expected and policy["max_calls"] == 12, "Permission mismatch")
    s.require(prefix == SYSTEM[locale] and raw == canonical(expected), "System prompt differs/private field leakage")
    return MARKER + raw


def check_projection(episode, expected_tools):
    row, spec = episode["row"], episode["spec"]
    permission(row, spec["policy"], episode["locale"])
    s.require(json.loads(row["tools"]) == expected_tools, "Tool schema changed")
    messages = row["conversations"]
    s.require(messages and messages[0]["from"] == "human" and messages[-1]["from"] == "gpt", "Role structure")
    for message in messages:
        s.require(set(message) == {"from", "value"} and message["from"] in ROLES
                  and isinstance(message["value"], str) and message["value"], "Private/invalid message fields")
        s.require(not re.search(r'"(?:fixture|state_seed|metric_eligible|fault_reads|pair_id|environment_audit)"\s*:', message["value"]),
                  "Hidden metadata in public messages")
    calls = [m for m in messages if m["from"] == "function_call"]
    if spec["category"] in {"deny_write", "no_tool"}:
        s.require(not calls and not episode["events"], "Tool-free gold contains calls")
    # The frozen validator independently checks each value's evidence origin.
    return {"id": episode["id"], "category": spec["category"], "variant": spec["variant"],
        "locale": episode["locale"], "allow_write": spec["policy"]["allow_write"],
        "missing_field": {"missing_email": "email", "email_followup": "email", "title_followup": "title", "priority_followup": "priority"}.get(spec["variant"])
            if spec["category"] == "clarify" else "email" if spec["variant"] == "missing_email" else None,
        "human_turns": sum(m["from"] == "human" for m in messages),
        "assistant_turns": sum(m["from"] in {"gpt", "function_call"} for m in messages),
        "tool_calls": len(episode["events"]), "first_assistant_role": next(m["from"] for m in messages if m["from"] in {"gpt", "function_call"}),
        "system_sha256": s.digest(row["system"]), "row_sha256": s.digest(row),
        "assistant_text_sha256": s.digest([m for m in messages if m["from"] in {"gpt", "function_call"}])}


def check_converted(row, converted):
    expected = [{"role": ROLES[m["from"]], "content": m["value"]} for m in row["conversations"]]
    s.require(converted["_prompt"] + converted["_response"] == expected, "Converter lost/changed roles or contents")
    s.require(converted["_system"] == row["system"] and converted["_tools"] == row["tools"], "Converter changed permission/tools")
    return expected


def verify_labels(pairs, ids, labels, turns, cutoff=4096):
    s.require(len(pairs) == turns, "Assistant turn dropped")
    full, wanted, spans = [], [], []
    for source, target in pairs:
        s.require(source and target, "Empty source/target")
        start = len(full) + len(source)
        full += list(source) + list(target)
        wanted += [-100] * len(source) + list(target)
        spans.append({"start": start, "end": len(full), "target_tokens": len(target)})
    s.require(ids == full and labels == wanted and len(full) <= cutoff, "Truncation or incorrect label masking")
    return {"total_tokens": len(full), "assistant_turns": turns, "supervised_tokens": sum(len(t) for _, t in pairs),
            "target_spans": spans, "input_ids_sha256": s.digest(ids), "labels_sha256": s.digest(labels), "passed": True}


def text_span(tokenizer, ids, text):
    # Offsets are derived from the REAL full-token sequence, not independent
    # substring tokenization (which can differ at BPE boundaries).
    decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    s.require(decoded.count(text) == 1, "Permission absent/ambiguous in actual source tokens")
    start, end = decoded.index(text), decoded.index(text) + len(text)
    def length(i):
        return len(tokenizer.decode(ids[:i], skip_special_tokens=False, clean_up_tokenization_spaces=False))
    def lower_bound(offset, strict=False):
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if (length(mid) > offset) if strict else (length(mid) >= offset):
                hi = mid
            else:
                lo = mid + 1
        return lo
    first = lower_bound(start, strict=True) - 1
    last = lower_bound(end)
    s.require(first >= 0 and length(first) <= start < length(first + 1)
              and length(last - 1) < end <= length(last), "Token offset boundary mismatch")
    s.require(text in tokenizer.decode(ids[first:last], skip_special_tokens=False, clean_up_tokenization_spaces=False),
              "Permission token span boundary cannot be verified")
    return {"start": first, "end": last, "text": text, "token_ids": ids[first:last],
            "token_ids_sha256": s.digest(ids[first:last]), "source_visible": True}


def match_training(run, input_audit, records):
    s.require(run["status"] == "complete" and run["stage"] == "train" and run["sources"] == s.predecessor.local_sources()
              and run["release_manifest_sha256"] == s.RELEASE_SHA and run["initializer_sha256"] == s.OLD_SHA,
              "Wrong actual training provenance")
    s.predecessor.check_config(run["config"], "train")
    expected = {"passed": True, "rows": len(records), "assistant_turns": sum(r["assistant_turns"] for r in records),
                "ordered_audit_sha256": s.digest(records)}
    s.require(input_audit == expected, "Actual Trainer input evidence differs from all-row token audit")
    return expected


def source_binding():
    manifest = s.read(s.RELEASE / "manifest.json")
    return {"release_manifest_sha256": s.sha(s.RELEASE / "manifest.json"),
            "release_files_sha256": manifest["files_sha256"], "candidate_manifest_sha256": s.sha(s.CANDIDATE / "manifest.json"),
            "episodes_sha256": s.sha(s.CANDIDATE / "episodes.json"),
            "config_sha256": s.sha(ROOT / "configs/ticket_sft_v3_train_v1_1.yaml"),
            "frozen_dev_sha256": s.DEV_SHA, "audit_sources": sources()}


def local_report_gate(report):
    s.require(report["status"] == "complete" and report["passed"] is True and report["rows"] == 3000,
              "Local audit incomplete")
    s.require(report["binding"] == source_binding(), "Local sources/release changed")
    s.require(report["uploads_sha256"] == UPLOADS, "Wrong diagnostic inputs")


def delivery_gate():
    manifest = s.read(OUT / "delivery.json")
    s.require(manifest["revision"] == "ticket-policy-audit-v1", "Wrong audit delivery")
    for name, expected in manifest["files_sha256"].items():
        p = Path(name)
        s.require(not p.is_absolute() and ".." not in p.parts and s.sha(ROOT / p) == expected, "Audit delivery file changed: " + name)
