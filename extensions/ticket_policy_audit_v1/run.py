"""local / remote / compare; no model weights, inference, optimizer or training."""
import argparse
from collections import Counter, defaultdict
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
from . import core as c
s = c.s


def local(directory, inputs):
    released = c.base_gate()
    import yaml
    s.predecessor.check_config(yaml.safe_load((c.ROOT / "configs/ticket_sft_v3_train_v1_1.yaml").read_text(encoding="utf-8")), "train")
    for name, expected in c.UPLOADS.items():
        s.require(s.sha(inputs / name) == expected, "Wrong uploaded diagnostic file: " + name)
    # Independently replay the exact reviewed diagnostic run in an isolated child.
    result = __import__("subprocess").run([sys.executable, "-B", "scripts/review_policy_diagnostic_uploads_v1.py",
        "--inputs", str(inputs.resolve()), "--output", str((directory / "diagnostic-review.json").resolve())],
        cwd=c.ROOT, capture_output=True, text=True, encoding="utf-8", timeout=600)
    (directory / "diagnostic-replay.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    s.require(result.returncode == 0, "Diagnostic replay failed; see retained log")
    snapshots = directory / "inputs"
    snapshots.mkdir()
    for name in c.UPLOADS:
        shutil.copyfile(inputs / name, snapshots / name)
    episodes = s.read(s.CANDIDATE / "episodes.json")
    train = s.read(s.RELEASE / "train.json")
    s.require(train == [e["row"] for e in episodes] and len(train) == 3000, "Published projection differs")
    records = [c.check_projection(e, c.TOOLS) for e in episodes]
    groups = defaultdict(list)
    for record in records:
        groups[(record["category"], record["variant"], record["locale"], record["allow_write"])].append(record)
    coverage = [{"category": key[0], "variant": key[1], "locale": key[2], "allow_write": key[3],
        "rows": len(rows), "tool_free_rows": sum(r["tool_calls"] == 0 for r in rows),
        "multi_user_rows": sum(r["human_turns"] > 1 for r in rows),
        "assistant_turns": sum(r["assistant_turns"] for r in rows),
        "unique_assistant_texts": len({r["assistant_text_sha256"] for r in rows})} for key, rows in sorted(groups.items())]
    tasks = c.diagnostic.build_tasks()
    interfaces = []
    for task in tasks:
        training_system = c.SYSTEM[task["locale"]] + "\n" + c.MARKER + c.canonical(task["execution_policy"])
        inference_system = c.diagnostic.system_prompt(task)
        interfaces.append({"id": task["id"], "profile": task["prompt_profile"], "locale": task["locale"],
            "request": task["request"], "policy": task["execution_policy"],
            "training_style_system": training_system, "diagnostic_system": inference_system,
            "system_equal": training_system == inference_system,
            "tools_equal": json.loads(train[0]["tools"]) == c.TOOLS})
    report = {"passed": True, "rows": 3000, "binding": c.source_binding(), "release_verification": released,
        "uploads_sha256": c.UPLOADS, "records": records, "coverage": coverage, "interfaces": interfaces,
        "diagnostic_review_sha256": s.sha(directory / "diagnostic-review.json"),
        "remote_tokenizer_audit": "pending", "actual_training_input_match": "pending_remote",
        "human_review": "pending", "training_started": False, "gpu_inference": False}
    (directory / "report.md").write_text(local_markdown(report), encoding="utf-8")
    return report


def local_markdown(report):
    lines = ["# 权限与监督审计：本地阶段", "", "全量发布数据与诊断重放检查通过；真实 tokenizer 和实际训练输入复核仍待服务器执行。不是完整根因结论，也不是人工批准。", "",
        "| 类别 | 变体 | 语言 | 写权限 | 条数 | 无调用 | 多轮用户 | 不同助手文本 |", "|---|---|---|---|---:|---:|---:|---:|"]
    for x in report["coverage"]:
        lines.append(f"| {x['category']} | {x['variant']} | {x['locale']} | {x['allow_write']} | {x['rows']} | {x['tool_free_rows']} | {x['multi_user_rows']} | {x['unique_assistant_texts']} |")
    lines += ["", "已证实：训练系统提示为双语 + JSON 权限，原推理为英文系统提示 + 权限句；schema 一致。训练拒写样例全部无调用，但回答文本高度模板化。JSON 诊断仍失败，不能据此证明提示格式就是根因。",
        "", "已有审计要求每个助手轮次监督，并在实际 Trainer 开始时逐条比较输入/标签哈希。本次远端将核对实际训练证据，再补充权限 token 位置。系统提示 label=-100 是正常的 prompt 屏蔽，不是信息不可见。",
        "", "本阶段没有加载权重、运行 GPU 推理、训练或读取旧最终测试；冻结 DEV 与旧评分未修改。"]
    return "\n".join(lines) + "\n"


def load_cpu_components(model_path, old_audit):
    s.require(sys.version_info[:2] == (3, 11), "Use fixed Python 3.11")
    versions = {k: importlib.metadata.version(k) for k in s.STACK}
    s.require(versions == s.STACK, "Pinned dependency versions differ")
    for name, expected in old_audit["tokenizer_asset_sha256"].items():
        s.require(s.sha(model_path / name) == expected, "Tokenizer asset changed: " + name)
    sys.meta_path.insert(0, c.CPUOnlyImports())
    # Import only AutoTokenizer and data/hparams modules, never model loader/chat/Trainer.
    from transformers import AutoTokenizer
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.converter import SharegptDatasetConverter
    from llamafactory.data.parser import DatasetAttr
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams import DataArguments
    for cls in (SharegptDatasetConverter, SupervisedDatasetProcessor):
        s.require(s.sha(inspect.getsourcefile(cls)) == old_audit["framework_source_sha256"][cls.__name__], "Preprocessor source changed")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path.resolve()), local_files_only=True, trust_remote_code=False,
                                             use_fast=True, padding_side="right", split_special_tokens=False)
    data_args = DataArguments(template="qwen", cutoff_len=4096, train_on_prompt=False, mask_history=False, packing=False)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    s.require(not template.efficient_eos, "Unexpected EOS policy")
    attr = DatasetAttr("file", "train.json")
    attr.join(s.read(s.RELEASE / "dataset_info.json")["ticket_sft_v3_train"])
    converter = SharegptDatasetConverter(attr, data_args)
    processor = SupervisedDatasetProcessor(template, tokenizer, None, data_args)
    return versions, tokenizer, template, converter, processor


def training_gate(old_records):
    directory = s.RUN / "train"
    path = directory / "run.json"
    s.require(s.sha(path) == c.TRAIN_RUN_SHA, "Wrong actual training run")
    run = s.read(path)
    for name, expected in run["outputs_sha256"].items():
        rel = Path(name)
        s.require(not rel.is_absolute() and ".." not in rel.parts and s.sha(directory / rel) == expected,
                  "Actual training output changed: " + name)
    actual = c.match_training(run, s.read(directory / "training-input-audit.json"), old_records)
    s.require(run["config_sha256"] == s.sha(c.ROOT / "configs/ticket_sft_v3_train_v1_1.yaml"), "Actual config changed")
    s.require(s.sha(directory / "adapter/adapter_model.safetensors") == c.diagnostic.NEW_SHA
              and s.sha(Path(s.OLD) / "adapter_model.safetensors") == s.OLD_SHA, "Adapter hash changed")
    verification = s.read(directory / "verification.json")
    s.require(verification["passed"] and verification["rows"] == 3000 and verification["global_step"] == 750
              and verification["epoch"] == 2 and verification["reload"]["finite_forward"], "Incomplete training acceptance")
    return {"run_sha256": s.sha(path), "training_input_audit_sha256": s.sha(directory / "training-input-audit.json"),
            "actual_training_inputs": actual, "weight_bytes_hashed_only": True}


def remote(directory, model_path):
    c.cpu_gate(os.environ)
    c.delivery_gate()
    c.base_gate()
    local_report = s.read(c.OUT / "local/report.json")
    c.local_report_gate(local_report)
    old = s.read(s.RELEASE / "token-audit.json")
    evidence = training_gate(old["records"])
    versions, tokenizer, template, converter, processor = load_cpu_components(model_path, old)
    episodes = s.read(s.CANDIDATE / "episodes.json")
    records = []
    with (directory / "records.jsonl").open("x", encoding="utf-8") as stream:
        for episode, prior in zip(episodes, old["records"]):
            row = episode["row"]
            text = c.permission(row, episode["spec"]["policy"], episode["locale"])
            converted = converter(row)
            messages = c.check_converted(row, converted)
            pairs = template.encode_multiturn(tokenizer, messages, row["system"], row["tools"])
            actual = processor.preprocess_dataset({k: [v] for k, v in converted.items()})
            s.require(len(actual["input_ids"]) == len(actual["labels"]) == 1, "Dropped row: " + episode["id"])
            ids, labels = actual["input_ids"][0], actual["labels"][0]
            turns = sum(m["from"] in {"gpt", "function_call"} for m in row["conversations"])
            metrics = {"id": episode["id"], **c.verify_labels(pairs, ids, labels, turns)}
            s.require(metrics == prior, "Fresh encoding differs from original audit: " + episode["id"])
            span = c.text_span(tokenizer, list(pairs[0][0]), text)
            s.require(all(v == -100 for v in labels[span["start"]:span["end"]]), "System labels unexpectedly supervised")
            record = {**metrics, "permission_span": span, "permission_labels_masked_as_expected": True,
                      "all_assistant_turns_supervised": True, "public_source_spans_masked": True}
            records.append(record)
            stream.write(c.canonical(record) + "\n")
            stream.flush()
            if len(records) % 100 == 0:
                print(f"CPU permission/token/label audit: {len(records)}/3000", flush=True)
    s.require(len(records) == 3000, "Audit incomplete")
    # Encode every initial diagnostic prompt using the exact HFProvider boundary.
    provider = s.legacy().HFProvider(tokenizer=tokenizer, template=template)
    uploaded_run = s.read(c.OUT / "local/inputs/run-ticket-sft-v3.json")
    interfaces = []
    for task, interface in zip(c.diagnostic.build_tasks(), local_report["interfaces"]):
        s.require(task["id"] == interface["id"], "Interface task order changed")
        info = provider.prompt_info(c.diagnostic.original_agent.input_payload(
            [{"role": "user", "content": task["request"]}], interface["diagnostic_system"]))
        s.require(info == uploaded_run["initial_prompt_audit"][task["id"]], "Diagnostic real token IDs differ: " + task["id"])
        encoded = {}
        for style, system in (("training_style", interface["training_style_system"]), ("diagnostic", interface["diagnostic_system"])):
            ids, _ = template.encode_oneturn(tokenizer, [{"role": "user", "content": task["request"]},
                {"role": "assistant", "content": ""}], system, c.canonical(c.TOOLS))
            ids = list(ids)
            s.require(len(ids) + 512 <= 8192, "Diagnostic context overflow")
            expected_text = (c.MARKER + c.canonical(task["execution_policy"])) if style == "training_style" else (
                c.MARKER + c.canonical({"allow_write": task["execution_policy"]["allow_write"]}) + "." if task["prompt_profile"] == "json_equivalent"
                else f"Write actions are {'permitted' if task['execution_policy']['allow_write'] else 'NOT permitted'} by the trusted client policy.")
            encoded[style] = {"tokens": len(ids), "input_ids_sha256": s.digest(ids),
                              "permission_span": c.text_span(tokenizer, ids, expected_text),
                              "decoded_prompt": tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)}
        interfaces.append({"id": task["id"], "profile": task["prompt_profile"], "locale": task["locale"], **encoded})
    s.write(directory / "interfaces.json", interfaces)
    import torch
    s.require(not torch.cuda.is_initialized(), "CPU audit unexpectedly initialized CUDA")
    return {"passed": True, "rows": 3000, "interfaces": 72, "binding": c.source_binding(),
        "local_report_sha256": s.sha(c.OUT / "local/report.json"), "versions": versions,
        "model_path": str(model_path.resolve()), "training_evidence": evidence,
        "records_sha256": s.sha(directory / "records.jsonl"), "interfaces_sha256": s.sha(directory / "interfaces.json"),
        "ordered_original_records_sha256": s.digest(old["records"]),
        "max_tokens": max(r["total_tokens"] for r in records), "all_turns": sum(r["assistant_turns"] for r in records),
        "cuda_initialized": False, "model_weights_loaded": False, "training_started": False, "human_review": "pending"}


def compare(directory):
    c.cpu_gate(os.environ)
    c.delivery_gate()
    c.base_gate()
    local_report = s.read(c.OUT / "local/report.json")
    c.local_report_gate(local_report)
    remote_path = c.OUT / "remote/report.json"
    remote_report = s.read(remote_path)
    s.require(remote_report["status"] == "complete" and remote_report["passed"] is True
              and remote_report["rows"] == 3000 and remote_report["interfaces"] == 72
              and remote_report["binding"] == local_report["binding"]
              and remote_report["local_report_sha256"] == s.sha(c.OUT / "local/report.json"), "Remote audit incomplete/unbound")
    s.require(remote_report["versions"] == s.STACK and remote_report["cuda_initialized"] is False
              and remote_report["model_weights_loaded"] is False and remote_report["training_started"] is False, "Remote audit scope changed")
    for filename, field in (("records.jsonl", "records_sha256"), ("interfaces.json", "interfaces_sha256")):
        s.require(s.sha(c.OUT / "remote" / filename) == remote_report[field], "Remote output hash differs")
    records = [json.loads(line) for line in (c.OUT / "remote/records.jsonl").read_text(encoding="utf-8").splitlines()]
    old_records = s.read(s.RELEASE / "token-audit.json")["records"]
    s.require(remote_report["training_evidence"] == training_gate(old_records), "Training evidence changed after remote audit")
    keys = set(old_records[0])
    s.require([{k: r[k] for k in keys} for r in records] == old_records, "Not all original records reproduced")
    for record in records:
        s.require(record["permission_span"]["source_visible"] and record["permission_labels_masked_as_expected"]
                  and record["all_assistant_turns_supervised"] and record["public_source_spans_masked"], "Incomplete per-row check")
    result = {"passed": True, "rows": 3000, "diagnostic_interfaces": 72, "local_report_sha256": s.sha(c.OUT / "local/report.json"),
        "remote_report_sha256": s.sha(remote_path), "human_review": "pending", "training_started": False,
        "confirmed": ["训练与推理系统提示存在差异，工具 schema 一致。",
                      "全部 3000 条在真实输入 token 中保留了可信权限，并正确监督所有助手轮次。",
                      "实际 Trainer 输入哈希与原审计及本次重新编码一致。",
                      "拒写标准轨迹无调用，回答文本多样性见本地覆盖统计。"],
        "excluded_in_this_pipeline": ["本次绑定链路中的权限丢失、预处理截断/助手轮次丢失、标签屏蔽错误和实际训练输入总体不符。"],
        "unknown": ["不安全行为的因果根因；本次数据审计不能证明优化强度是否足够或模型如何迁移到不同提示。"],
        "recommendations": ["调用前拒写：独立变化可信权限与用户意图，始终保留执行器授权门禁。",
                            "澄清：覆盖邮箱稍后补充和多轮续接，不填写空占位符、不索取内部 ID。",
                            "忠实返回：覆盖被拦截和部分成功，不虚构创建成功或未查询。",
                            "优先级：明确 high → normal → low，提供通用例子，不编造 SLA。",
                            "任何新数据或训练另行批准；冻结 DEV 和旧最终测试不变。"]}
    text = "# 权限可见性与监督链路：最终审计\n\n全量链路审计通过，不等于模型行为验收通过，也不是人工批准。\n\n"
    for title, field in (("已证实", "confirmed"), ("本链路中已排除", "excluded_in_this_pipeline"), ("仍无法确定", "unknown"), ("修复建议（未执行）", "recommendations")):
        text += "## " + title + "\n\n" + "\n".join("- " + item for item in result[field]) + "\n\n"
    text += "权限文字在输入中可见、系统标签为 -100 是正常机制。当前结果更支持行为学习/迁移不足这一待验证方向，不能直接证明其训练根因。JSON 诊断仍失败，因此不能只靠提示格式宣布修复。\n\n逐条 ID、token 位置、输入与标签哈希见 remote/records.jsonl；72 个成对提示全文与 token 哈希见 remote/interfaces.json。所有数据和旧评分保持不变；未读取旧最终测试。\n"
    (directory / "report.md").write_text(text, encoding="utf-8")
    return result


def execute(args):
    directory = c.OUT / args.stage
    s.fresh(directory)
    directory.mkdir(parents=True)
    started = {"status": "started", "stage": args.stage, "started_at_utc": s.now(),
               "training_started": False, "gpu_inference": False, "human_review": "pending"}
    s.write(directory / "status.json", started)
    try:
        if args.stage == "local":
            s.require(args.inputs is not None, "local requires --inputs")
            result = local(directory, args.inputs)
        elif args.stage == "remote":
            result = remote(directory, args.model_path)
        else:
            result = compare(directory)
        result.update(status="complete", stage=args.stage, finished_at_utc=s.now())
        s.write(directory / "report.json", result)
        s.update(directory / "status.json", {**started, "status": "complete", "report_sha256": s.sha(directory / "report.json"), "finished_at_utc": s.now()})
        print(json.dumps({"stage": args.stage, "passed": True, "rows": result["rows"], "training_started": False,
                          "remote_audit": "pending" if args.stage == "local" else "completed"}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        s.update(directory / "status.json", {**started, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_at_utc": s.now()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("local", "remote", "compare"))
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--model-path", type=Path, default=Path(s.MODEL))
    execute(parser.parse_args())
