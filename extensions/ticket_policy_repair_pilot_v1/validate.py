"""Independent tool/state replay and public evidence/claim checks for the pilot."""
import copy
import json
from collections import Counter
from . import core as c
s = c.s


def forbidden():
    dev = s.frozen_descriptor()
    ids = set(dev["identifiers"])
    requests = {t["request"] for t in dev["tasks"]}
    old = s.read(s.CANDIDATE / "episodes.json")
    for e in old:
        requests.add(e["row"]["conversations"][0]["value"])
        for table in ("users", "tickets", "articles"):
            for row in e["spec"]["fixture"][table]:
                ids.update(str(row[k]) for k in ("user_id", "ticket_id", "article_id", "email") if k in row)
    for task in c.audit.diagnostic.build_tasks():
        requests.add(task["request"])
        state = c.audit.diagnostic.fixture(task["state_seed"])
        for table in ("users", "tickets", "articles"):
            for row in state[table]:
                ids.update(str(row[k]) for k in ("user_id", "ticket_id", "article_id", "email") if k in row)
    return {"ids": ids, "requests": requests, "frozen_dev_sha256": s.DEV_SHA,
            "old_train_sha256": s.sha(s.RELEASE / "train.json"), "diagnostic_tasks_sha256": s.digest(c.audit.diagnostic.build_tasks())}


def observe(response, users, tickets):
    if not response["ok"]: return
    rows = response["result"] if isinstance(response["result"], list) else [response["result"]]
    for row in rows:
        if isinstance(row, dict):
            if "user_id" in row and "email" in row: users[row["user_id"]] = copy.deepcopy(row)
            if "ticket_id" in row: tickets[row["ticket_id"]] = copy.deepcopy(row)


def claim_check(claim, spec, human, users, tickets, history):
    kind = claim["kind"]
    if kind == "denied":
        s.require(not spec["policy"]["allow_write"] and not history, "Refusal after calls or while authorized")
    elif kind == "missing":
        field = claim["field"]
        s.require(spec["policy"]["allow_write"] and not history and field in {"email", "title"}
                  and spec["fields"][field] not in human, "Incorrect/mistimed clarification")
    elif kind == "ticket":
        t = claim["ticket"]
        s.require(t["ticket_id"] in tickets and all(tickets[t["ticket_id"]][k] == v for k, v in t.items()), "Unsupported ticket success/state")
    elif kind == "closed":
        t = tickets.get(claim["id"])
        s.require(t and t["status"] == "closed" and t["assignee_id"] == claim["assignee_id"]
                  and not any(e["call"]["name"] in c.sim.WRITE_TOOLS and e["call"]["arguments"].get("ticket_id") == claim["id"] for e in history),
                  "Closed ticket changed or skipped without read evidence")
    elif kind in {"not_found", "no_requester"}:
        name, field, value, code = (("get_ticket", "ticket_id", claim["id"], "ticket_not_found") if kind == "not_found"
            else ("lookup_user", "email", claim["email"], "user_not_found"))
        s.require(any(e["call"]["name"] == name and e["call"]["arguments"][field] == value and not e["response"]["ok"]
                      and e["response"]["error"]["code"] == code for e in history), "Missing not-found evidence")
        s.require(not any(e["call"]["name"] in c.sim.WRITE_TOOLS for e in history), "Wrote after not-found")
    elif kind == "empty":
        s.require(any(e["call"] == {"name": "search_kb", "arguments": {"query": claim["query"]}} and e["response"]["ok"]
                      and e["response"]["result"] == [] for e in history), "Missing empty-result evidence")
    else:
        s.require(kind == "priority" and not history and claim["order"] == ["high", "normal", "low"]
                  and claim["sla"] is None and claim["examples"] == (spec["variant"] != "order"), "Incorrect priority/example/SLA claim")


def validate_episode(e, blocked):
    spec, row = e["spec"], e["row"]
    s.require(spec == c.specs()[spec["index"] - 1] and e["id"] == spec["pair_id"] + "-" + e["locale"], "Spec/ID changed")
    s.require(set(row) == {"system", "tools", "conversations"} and row["system"] == c.system(spec, e["locale"])
              and json.loads(row["tools"]) == c.TOOLS, "Hidden metadata/system/schema changed")
    text = c.audit.canonical(row)
    # Tokenize identifiers rather than doing thousands of substring scans.
    import re
    public_ids = set(re.findall(r'(?:U|T|KB)[A-Z0-9]{6,}|[A-Za-z0-9._+-]+@example\.test', text))
    s.require(not public_ids & blocked["ids"] and row["conversations"][0]["value"] not in blocked["requests"], "Copied DEV/old training/diagnostic identity or request")
    messages = row["conversations"]
    s.require(messages[0] == {"from": "human", "value": c.request(spec, e["locale"])} and messages[-1]["from"] == "gpt"
              and len(messages) % 2 == 0, "Invalid opening/final/roles")
    users, tickets, history, human, cursor = {}, {}, [], "", 0
    claims = {x["index"]: x for x in e["claims"]}
    with c.sim.Environment(spec) as env:
        s.require(env.snapshot() == e["initial_state"], "Initial database mismatch")
        for i, m in enumerate(messages):
            s.require(set(m) == {"from", "value"} and isinstance(m["value"], str) and m["value"], "Invalid public message")
            s.require(m["from"] in (("human", "observation") if i % 2 == 0 else ("gpt", "function_call")), "Role alternation")
            if m["from"] == "human":
                human += "\n" + m["value"]
            elif m["from"] == "function_call":
                calls = json.loads(m["value"])
                s.require(len(calls) == 1 and messages[i + 1]["from"] == "observation", "Missing observation/invalid batch")
                call = calls[0]
                name, args = call["name"], call["arguments"]
                s.require(set(call) == {"name", "arguments"} and name in env.validators
                          and not list(env.validators[name].iter_errors(args)), "Invalid tool/schema")
                s.require(spec["policy"]["allow_write"] or name not in c.sim.WRITE_TOOLS, "Unauthorized write")
                for k, v in args.items():
                    if k == "requester_id": s.require(v in users, "Unobserved requester")
                    elif k == "expected_version": s.require(tickets.get(args["ticket_id"], {}).get("version") == v, "Unobserved/stale version")
                    elif k == "ticket_id": s.require(v in human or v in tickets, "Unobserved ticket")
                    else: s.require(str(v) in human, "Argument not provided by user: " + k)
                event = e["events"][cursor]
                key = f'{spec["pair_id"]}:{cursor + 1}' if name in c.sim.WRITE_TOOLS else None
                s.require(call == event["call"] and event["key"] == key and env.snapshot() == event["before"], "Call/key/state mismatch")
                response = env.execute(name, args, idempotency_key=key)
                s.require(response == event["response"] and json.loads(messages[i + 1]["value"]) == [response]
                          and env.snapshot() == event["after"], "Tool response not from independent execution")
                if not response["ok"]:
                    s.require(response["error"]["code"] in {"user_not_found", "ticket_not_found"}, "Invalid gold action")
                observe(response, users, tickets)
                history.append(event)
                cursor += 1
            elif m["from"] == "observation":
                s.require(messages[i - 1]["from"] == "function_call", "Orphan observation")
            else:
                entry = claims[i]
                for claim in entry["claims"]: claim_check(claim, spec, human, users, tickets, history)
                s.require(m["value"] == "\n".join(c.render(x, e["locale"], entry["style"]) for x in entry["claims"]), "Answer differs from evidence-backed claims")
        s.require(cursor == len(e["events"]) <= 12 and env.snapshot() == e["final_state"]
                  and env.audit == e["environment_audit"] and env.external_events == e["external_events"], "Final state/audit/budget differs")
    if spec["category"] == "priority" or spec["variant"] in {"create_denied", "override_denied", "assign_denied", "permission_before_missing", "email_stop"}:
        s.require(cursor == 0 and e["initial_state"] == e["final_state"], "Required zero-call example changed state")
    s.require(c.generate(spec, e["locale"]) == e, "Deterministic goal reconstruction differs")
    return {"id": e["id"], "passed": True, "calls": cursor, "state_changed": e["initial_state"] != e["final_state"],
            "assistant_turns": sum(m["from"] in {"gpt", "function_call"} for m in messages)}


def validate_population(episodes, blocked):
    s.require(len(episodes) == 36 and len({e["id"] for e in episodes}) == 36, "Wrong pilot population")
    s.require(Counter(e["spec"]["category"] for e in episodes) == {"authority": 12, "clarify": 8, "faithful": 10, "priority": 6}, "Wrong category quota")
    s.require([e["id"] for e in episodes] == [spec["pair_id"] + "-" + locale for spec in c.specs() for locale in ("zh", "en")], "Bilingual order differs")
    records = [validate_episode(e, blocked) for e in episodes]
    for zh, en in zip(episodes[::2], episodes[1::2]):
        for field in ("spec", "events", "claims", "initial_state", "final_state", "environment_audit", "external_events"):
            s.require(zh[field] == en[field], "Bilingual semantic/trace mismatch: " + field)
    for group in ("p1", "p2", "p3"):
        pair = [e for e in episodes if e["spec"]["contrast_group"] == group and e["locale"] == "zh"]
        a, b = pair
        s.require(a["row"]["conversations"][0] == b["row"]["conversations"][0] and a["initial_state"] == b["initial_state"]
                  and a["spec"]["profile"] == b["spec"]["profile"]
                  and a["spec"]["policy"]["allow_write"] is False and b["spec"]["policy"]["allow_write"] is True,
                  "Counterfactual changed request/database/profile")
        s.require(not a["events"] and b["events"], "Contrast does not show correct policy-dependent behavior")
    return {"passed": True, "rows": 36, "bilingual_pairs": 18, "counterfactual_groups": 3, "records": records,
            "real_model_evaluated": False, "real_tokenizer_audited": False, "ready_for_training": False,
            "human_review": "pending", "bulk_generated": False, "training_started": False}
