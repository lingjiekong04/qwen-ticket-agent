"""GPU1/offline inference without distributed env; preserve original v1 failure."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.ticket_policy_diagnostic_v1_1 import core as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "prepare", "hermes-r8", "ticket-sft-v3", "compare"), default="all")
    args = parser.parse_args()
    c.delivery_gate()
    if args.stage == "all":
        c.s.fresh(c.OUTPUT)
        c.s.fresh(c.LOGS)
    environment, removed = c.child_environment(os.environ)
    c.environment_gate(environment)
    print("Single-process child: distributed variables removed:", removed,
          "; CUDA_VISIBLE_DEVICES=1; WORLD_SIZE/RANK/LOCAL_RANK absent", flush=True)
    stages = ("prepare", "hermes-r8", "ticket-sft-v3", "compare") if args.stage == "all" else (args.stage,)
    c.LOGS.mkdir(exist_ok=True)
    for stage in stages:
        log = c.LOGS / f"{stage}.log"
        status_path = c.LOGS / f"{stage}.json"
        for path in (log, status_path): c.s.fresh(path)
        command = [sys.executable, "-u", "-B", "-m", "extensions.ticket_policy_diagnostic_v1_1.run", "--stage", stage]
        status = {"stage": stage, "status": "started", "command": command, "sources": c.sources(),
                  "started_at_utc": datetime.now(timezone.utc).isoformat(), "removed_environment_keys": removed,
                  "controlled_environment": {key: environment[key] for key in ("CUDA_VISIBLE_DEVICES", "FORCE_TORCHRUN", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")},
                  "distributed_environment_absent": True, "training_started": False}
        c.s.write(status_path, status)
        print("Launching:", command, flush=True)
        try:
            with log.open("x", encoding="utf-8") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, encoding="utf-8")
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            status["exit_code"] = code
            c.s.require(code == 0, f"Diagnostic failed (exit={code}); preserve outputs/log: {log}")
            status["status"] = "complete"
        except BaseException as exc:
            # On interruption do not leave a detached GPU child behind. This is
            # only the subprocess started by this launcher, never another task.
            if "process" in locals() and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
            status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            if log.exists(): status["log_sha256"] = c.s.sha(log)
            c.s.update(status_path, status)
    print("Repaired inference diagnostic completed. No training started; semantics/human approval remain pending.")


if __name__ == "__main__": main()
