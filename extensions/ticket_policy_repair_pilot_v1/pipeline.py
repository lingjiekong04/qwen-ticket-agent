"""Generate/replay/review a small pilot only. No bulk, release or train command."""
import argparse
import html
from pathlib import Path
import json
from . import core as c
from .validate import forbidden, validate_population
s = c.s


def review(episodes):
    lines = ["# 修复代表样例：18 对 / 36 条", "", "AI 辅助设计，待确认；不是人工批准、训练发布或模型成绩。所有工具返回由独立 SQLite 模拟执行产生。", "",
        "3 组权限对照保持相同请求和初始数据库，只改变可信权限；3 种提示形式用于覆盖，不代表任何提示已修复模型。", ""]
    cards = []
    for a, b in zip(episodes[::2], episodes[1::2]):
        spec = a["spec"]
        heading = f'{spec["pair_id"]} · {spec["category"]}/{spec["variant"]} · {spec["profile"]} · allow_write={spec["policy"]["allow_write"]}'
        lines += ["## " + heading, ""]
        body = []
        for label, e in (("中文", a), ("English", b)):
            messages = e["row"]["conversations"]
            calls = " → ".join(x["call"]["name"] + (" ✓" if x["response"]["ok"] else " ✗ " + x["response"]["error"]["code"]) for x in e["events"]) or "无调用 / no calls"
            req = messages[0]["value"]
            turns = "\n".join(m["from"] + ": " + m["value"] for m in messages if m["from"] in {"human", "gpt"})
            lines += [f"### {label}", "", "请求：" + req, "", "调用链：" + calls, "", "```text", turns, "```", ""]
            body.append(f'<h3>{label}</h3><p>{html.escape(req)}</p><p>调用链：{html.escape(calls)}</p><pre>{html.escape(turns)}</pre>')
        changed = a["initial_state"] != a["final_state"]
        lines += ["状态变化：" + ("数据库发生请求范围内的修改。" if changed else "数据库未修改。"), ""]
        states = {"initial": a["initial_state"], "final": a["final_state"]}
        body += [f'<p>状态变化：{changed}</p><details><summary>完整状态变化</summary><pre>{html.escape(json.dumps(states, ensure_ascii=False, indent=2))}</pre></details>',
                 f'<details><summary>完整中英 episode JSON（包含验证元数据；不进入模型输入）</summary><pre>{html.escape(json.dumps([a,b],ensure_ascii=False,indent=2))}</pre></details>']
        cards.append(f'<section><h2>{html.escape(heading)}</h2>{"".join(body)}</section>')
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>修复代表样例</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px}section{border-top:1px solid #999;padding:20px 0}pre{white-space:pre-wrap;word-break:break-word;background:#f5f5f5;padding:12px}details{margin:12px 0}</style><h1>修复代表样例：18 对 / 36 条</h1><p>待确认，不可直接训练；AI 辅助检查不计作人工批准。仅模拟工具执行，没有模型推理。</p>' + ''.join(cards) + '</html>'
    return "\n".join(lines), page


def build():
    c.audit.delivery_gate()
    c.audit.base_gate()
    s.fresh(c.OUTPUT)
    c.OUTPUT.mkdir(parents=True)
    s.write(c.OUTPUT / "status.json", {"status": "started", "revision": c.REVISION, "training_started": False})
    try:
        episodes = [c.generate(spec, locale) for spec in c.specs() for locale in ("zh", "en")]
        s.write(c.OUTPUT / "episodes.json", episodes)
        blocked = forbidden()
        validation = validate_population(episodes, blocked)
        s.write(c.OUTPUT / "validation.json", validation)
        md, page = review(episodes)
        (c.OUTPUT / "review.md").write_text(md, encoding="utf-8")
        (c.OUTPUT / "review.html").write_text(page, encoding="utf-8")
        s.write(c.OUTPUT / "manifest.json", {"revision": c.REVISION, "seed": c.SEED, "rows": 36, "pairs": 18,
            "sources": c.source_hashes(), "simulator_source_sha256": c.SOURCE_SHA,
            "isolation": {k: blocked[k] for k in ("frozen_dev_sha256", "old_train_sha256", "diagnostic_tasks_sha256")},
            "files_sha256": {name: s.sha(c.OUTPUT / name) for name in ("episodes.json", "validation.json", "review.md", "review.html")},
            "ready_for_training": False, "bulk_generated": False, "training_started": False,
            "real_model_evaluated": False, "real_tokenizer_audited": False, "human_review": "pending", "confirmation": "pending"})
        s.update(c.OUTPUT / "status.json", {"status": "complete", "manifest_sha256": s.sha(c.OUTPUT / "manifest.json"), "training_started": False})
        print(json.dumps({k: validation[k] for k in validation if k != "records"}, ensure_ascii=False))
    except BaseException as exc:
        s.update(c.OUTPUT / "status.json", {"status": "failed", "error": repr(exc), "training_started": False})
        raise


def verify():
    manifest = s.read(c.OUTPUT / "manifest.json")
    s.require(manifest["sources"] == c.source_hashes() and manifest["ready_for_training"] is False
              and manifest["confirmation"] == "pending" and manifest["human_review"] == "pending", "Source/gate changed")
    for name, expected in manifest["files_sha256"].items():
        s.require(Path(name).name == name and s.sha(c.OUTPUT / name) == expected, "Artifact changed")
    result = validate_population(s.read(c.OUTPUT / "episodes.json"), forbidden())
    s.require(result == s.read(c.OUTPUT / "validation.json"), "Replayed validation differs")
    print(json.dumps({"passed": True, "rows": 36, "pairs": 18, "ready_for_training": False, "human_review": "pending"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args()
    build() if args.command == "build" else verify()
