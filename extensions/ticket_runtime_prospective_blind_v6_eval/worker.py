"""GPU worker for v6. It cannot import or read the oracle or scorer."""

from __future__ import annotations

import json
import os

from extensions.ticket_runtime_model_recovery_v1 import provider as providers
from extensions.ticket_runtime_model_recovery_v1 import support as model_support
from . import support as s


def execute() -> dict:
    model_support.check_environment(dict(os.environ), gpu=True)
    tasks = s.worker_source_gate()
    s.weights_gate(server=True)
    gpu = model_support.gpu_idle_gate()
    from llamafactory.chat import ChatModel
    from transformers import set_seed

    model = ChatModel(s.model_args())
    providers.check_loaded(model, gpu["uuid"])
    provider = providers.HFProvider(model=model)
    streams = {arm: (s.RUN / arm / "episodes-unscored.jsonl").open(
        "x", encoding="utf-8", newline="\n") for arm in s.ARMS}
    try:
        for index, task in enumerate(tasks):
            for arm in s.ARMS:
                set_seed(2026102207 + index)
                record = s.runtime_for_arm(arm).run_episode(
                    task, provider,
                    variant="ticket-policy-retention-v1.2-gpu2")
                s.require(record["task_sha256"] == s.runtime_v1.digest(task) and
                          record["scores"] is None and
                          not record["oracle_loaded_during_generation"],
                          "Unscored v6 worker record invalid")
                streams[arm].write(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                              separators=(",", ":"), allow_nan=False) + "\n")
                streams[arm].flush()
                print(f"{index + 1}/160 {task['id']} {arm}: {record['termination']}", flush=True)
    finally:
        for stream in streams.values():
            stream.close()
    return {"passed": True, "episodes": 320,
            "arms": {arm: {"episodes": 160,
                            "unscored_sha256": s.sha(s.RUN / arm / "episodes-unscored.jsonl")}
                     for arm in s.ARMS},
            "oracle_loaded": False, "training_started": False, "live_tools_executed": 0}


if __name__ == "__main__":
    print(json.dumps(execute(), ensure_ascii=False, indent=2), flush=True)
