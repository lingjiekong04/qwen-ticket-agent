"""Human-confirmed contracts, persistent simulated sessions and v1.7 execution.

No holdout or evaluation oracle is imported. The frozen episode/guard/parser and
renderer perform the work. Only the execution driver and session persistence are
new: the single-episode backend's creation counter is carried across requests.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import uuid
from pathlib import Path

from extensions.ticket_runtime_contract_guard_integration_v1 import core as wire
from extensions.ticket_runtime_contract_guard_v1_1 import core as guard
from extensions.ticket_runtime_trusted_completion_v1_7 import core as runtime

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OPERATIONS = {
    "get_ticket": ("ticket_id",),
    "update_ticket_status": ("ticket_id", "status"),
    "assign_ticket": ("ticket_id", "assignee_id"),
    "create_ticket": ("title", "description", "priority", "requester_id"),
    "lookup_user": ("email",),
    "search_tickets": ("query", "requester_id", "status"),
    "knowledge": ("query",),
}
TICKET_FIELDS = ["ticket_id", "status", "version", "priority", "assignee_id"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fixture() -> dict:
    return load(ROOT / "data/ticket_interactive_demo_v1/fixture.json")


def build_task(body: dict, state: dict) -> dict:
    require(isinstance(body, dict) and set(body) == {
        "operation", "request", "parameters", "confirm_contract", "allow_write"},
        "请求字段必须为 operation/request/parameters/confirm_contract/allow_write")
    operation, parameters = body["operation"], body["parameters"]
    require(isinstance(operation, str) and operation in OPERATIONS, "不支持的操作")
    require(body["confirm_contract"] is True, "请先确认可信参数")
    require(type(body["allow_write"]) is bool, "allow_write 必须为布尔值")
    require(isinstance(body["request"], str) and 1 <= len(body["request"].strip()) <= 4000,
            "请输入 1–4000 字符的请求")
    require(isinstance(parameters, dict) and set(parameters) == set(OPERATIONS[operation]),
            "操作参数缺失或包含额外字段")
    for key, value in parameters.items():
        maximum = 2000 if key == "description" else 120 if key == "title" else 200
        require(isinstance(value, str) and 0 < len(value) <= maximum and value.strip() == value,
                "参数不能为空、不能含首尾空格，且必须在长度限制内：" + key)
    if "status" in parameters:
        require(parameters["status"] in guard.STATUSES, "状态值无效")
    if "priority" in parameters:
        require(parameters["priority"] in guard.PRIORITIES, "优先级无效")
    writes = operation in {"create_ticket", "assign_ticket", "update_ticket_status"}
    require(not writes or body["allow_write"], "本次模拟写入需要勾选授权")
    # Contract fields come from the human form, never from model output or prose.
    p = copy.deepcopy(parameters)
    if operation == "get_ticket":
        workflow = guard.TrustedWorkflow("read_ticket", locale="zh", target_ticket_id=p["ticket_id"])
    elif operation == "update_ticket_status":
        workflow = guard.TrustedWorkflow("update_status", locale="zh", target_ticket_id=p["ticket_id"],
                                         target_status=p["status"])
    elif operation == "assign_ticket":
        workflow = guard.TrustedWorkflow("assign_ticket", locale="zh", target_ticket_id=p["ticket_id"],
                                         target_assignee_id=p["assignee_id"])
    elif operation == "create_ticket":
        workflow = guard.TrustedWorkflow("create_ticket", locale="zh", requester_id=p["requester_id"],
                                         create_payload={key: p[key] for key in ("title", "description", "priority")})
    elif operation == "lookup_user":
        workflow = guard.TrustedWorkflow("lookup_user", locale="zh", email=p["email"])
    elif operation == "search_tickets":
        workflow = guard.TrustedWorkflow("search_tickets", locale="zh", search_filters=p)
    else:
        workflow = guard.TrustedWorkflow("knowledge", locale="zh", kb_query=p["query"], max_articles=2)
    target = workflow.target_ticket_id
    if target:
        ticket = next((x for x in state["tickets"] if x["ticket_id"] == target), None)
        require(ticket is not None, "模拟库中没有该工单 ID")
        if operation == "update_ticket_status":
            require(wire.design.ALLOWED_NEXT.get(ticket["status"]) == p["status"],
                    "仅支持 open → in_progress → resolved → closed 的相邻迁移")
    if operation == "assign_ticket":
        require(any(x["agent_id"] == p["assignee_id"] for x in state["agents"]), "模拟库中没有该负责人")
    if operation == "create_ticket":
        require(any(x["user_id"] == p["requester_id"] for x in state["users"]), "请使用模拟库内的 requester_id")
    scope = workflow.scope()
    # Appended JSON makes every exact value visible to Qwen and recovery guards.
    visible = wire.canonical({"operation": operation, "parameters": p,
                              "write_authorized": writes and body["allow_write"]})
    task = {
        "id": "demo-" + uuid.uuid4().hex,
        "family": operation, "primary_tool": operation,
        "locale": "zh", "profile": "json", "entry_kind": "interactive_direct",
        "request": (body["request"].strip() + "\n\n用户确认的执行契约（以本段为准）：" + visible +
                    "\n先核实证据再执行，只返回可核实结果。"),
        "request_scope": scope, "trusted_write_allowed": writes and body["allow_write"],
        "initial_state": copy.deepcopy(state), "fault_plan": None,
        "answer_contract": {"required_entity_ids": scope["target_ticket_ids"],
                            "required_answer_fields": TICKET_FIELDS if target else [], "output_profile": "json"},
    }
    fields = {
        "target_status": workflow.target_status, "target_assignee_id": workflow.target_assignee_id,
        "trusted_email": workflow.email, "trusted_requester_id": workflow.requester_id,
        "trusted_create_payload": workflow.create_payload, "trusted_search_filters": workflow.search_filters,
        "trusted_kb_query": workflow.kb_query, "max_articles": workflow.max_articles,
    }
    task.update({key: value for key, value in fields.items() if value is not None})
    wire.workflow_from_task(task)
    return task


def run_task(task: dict, provider) -> dict:
    """Use frozen episode decisions while safely carrying mock state across turns."""
    episode = runtime.TrustedCompletionEpisode(task)
    numbers = [int(key[1:]) for key in episode.backend.tickets if re.fullmatch(r"N\d+", key)]
    episode.backend.next_id = max([999] + numbers) + 1
    messages = [{"role": "user", "content": task["request"]}]
    rounds, termination, error = [], "generation_budget_exceeded", None
    try:
        for index in range(1, wire.MAX_GENERATIONS + 1):
            payload = wire.payload(messages, task)
            info = provider.prompt_info(copy.deepcopy(payload))
            generated = provider.generate(copy.deepcopy(payload))
            rounds.append({"index": index, "prompt_info": info, "response": copy.deepcopy(generated)})
            processed = episode.process(generated["text"])
            messages.append({"role": "assistant", "content": generated["text"]})
            if processed.get("released"):
                termination = "safe_stop" if processed["kind"] == "safe_stop" else "final_response"
                break
            if processed["kind"] == "calls":
                for item in processed["observations"]:
                    call = item["response"].get("controller_recovery_call") or item["call"]
                    wire.append_observation(messages, call, item["response"])
                if episode.guard.halted:
                    termination = "guard_halted"
                    break
            else:
                messages.append({"role": "user", "content": wire.canonical(processed["feedback"])})
    except Exception as exc:
        # Preserve the actual committed mock state and every available event.
        termination, error = "provider_or_runtime_error", type(exc).__name__ + ": " + str(exc)[:500]
    after = copy.deepcopy(episode.backend.tickets)
    before = episode.before
    changes = [{"ticket_id": key, "before": before.get(key), "after": after.get(key)}
               for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]
    return {
        "request_id": task["id"], "runtime": "trusted_completion_v1.7", "termination": termination,
        "completed": termination == "final_response" and bool(episode.released_text),
        "final_text": episode.released_text or "", "final_source": (
            "trusted_contract_evidence_renderer" if termination == "final_response" else "safe_stop_or_no_release"),
        "error": error, "rounds": rounds, "events": copy.deepcopy(episode.events),
        "answers": copy.deepcopy(episode.answers), "invalid_outputs": episode.invalid_outputs,
        "before": before, "after": after, "changes": changes,
        "recovery": copy.deepcopy(episode.recovery), "model_rounds": len(rounds),
        "live_tools_executed": 0, "task_sha256": wire.digest(task),
    }


class DemoService:
    def __init__(self, provider, *, provider_kind: str, audit_dir: Path):
        require(provider_kind in {"scripted", "local-qwen"}, "Provider kind invalid")
        self.provider, self.provider_kind = provider, provider_kind
        self.sessions, self.lock, self.rolled_back = {}, threading.RLock(), False
        audit_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = audit_dir / ("session-" + uuid.uuid4().hex + ".jsonl")
        # Fail before admitting requests if audit storage is not writable.
        with self.audit_path.open("x", encoding="utf-8"):
            pass

    def new_session(self) -> dict:
        with self.lock:
            require(len(self.sessions) < 64, "演示会话数量达到 64，请重启演示")
            key = uuid.uuid4().hex
            self.sessions[key] = {"state": fixture(), "requests": {}}
            return {"session_id": key, "state": copy.deepcopy(self.sessions[key]["state"])}

    def get_state(self, session_id: str) -> dict:
        with self.lock:
            require(session_id in self.sessions, "会话不存在，请创建新会话")
            return copy.deepcopy(self.sessions[session_id]["state"])

    def rollback(self) -> dict:
        with self.lock:
            self.rolled_back = True
            return {"state": "manual_safe_stop", "live_tools_executed": 0}

    def handle(self, session_id: str, request_key: str, body: dict) -> dict:
        # Inference and snapshot commits are serialized, including rollback.
        with self.lock:
            require(session_id in self.sessions, "会话不存在")
            require(isinstance(request_key, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", request_key),
                    "请求键格式无效")
            session = self.sessions[session_id]
            fingerprint = wire.digest(body)
            previous = session["requests"].get(request_key)
            if previous:
                require(previous["fingerprint"] == fingerprint, "同一请求键不能对应不同内容")
                return {**copy.deepcopy(previous["response"]), "replayed": True}
            require(not self.rolled_back, "已安全停机，请重启演示后继续")
            require(len(session["requests"]) < 100, "会话已达 100 次请求，请创建新会话")
            task = build_task(body, session["state"])
            response = run_task(task, self.provider)
            response.update(provider=self.provider_kind, session_id=session_id,
                            visible_request=task["request"], replayed=False)
            response["audit_sha256"] = wire.digest(response)
            next_state = copy.deepcopy(session["state"])
            next_state["tickets"] = list(copy.deepcopy(response["after"]).values())
            # Cache the committed state and replay before appending evidence so
            # a disk failure cannot make a retry create another simulated write.
            session["state"] = next_state
            session["requests"][request_key] = {"fingerprint": fingerprint, "response": copy.deepcopy(response)}
            with self.audit_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(wire.canonical(response) + "\n")
                stream.flush()
            return response


class SyntheticProvider:
    """Explicit CPU test double. Never used as fallback for local Qwen."""
    def prompt_info(self, payload):
        return {"kind": "scripted_cpu", "sha256": wire.digest(payload)}

    def generate(self, payload):
        text = payload["messages"][0]["content"]
        contract = json.JSONDecoder().raw_decode(text.split("用户确认的执行契约（以本段为准）：", 1)[1])[0]
        operation, p = contract["operation"], contract["parameters"]
        calls = []
        if operation in {"update_ticket_status", "assign_ticket"}:
            observations = [x for x in payload["messages"] if x["role"] == "observation"]
            if not observations:
                calls = [guard.call("get_ticket", ticket_id=p["ticket_id"])]
            else:
                observation = json.loads(observations[-1]["content"])[0]["response"]
                version = observation["result"]["version"]
                calls = [guard.call(operation, **p, expected_version=version)]
        elif operation == "knowledge":
            calls = [guard.call("search_kb", query=p["query"])]
        else:
            calls = [guard.call(operation, **p)]
        return {"text": "\n".join("<tool_call>" + wire.canonical(x) + "</tool_call>" for x in calls),
                "finish_reason": "scripted", "prompt_tokens": 0, "response_tokens": 0}
