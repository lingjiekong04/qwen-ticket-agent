"""Recompare completed smoke and finish frozen DEV. No training entry point."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.ticket_training_v1_2 import support as s
from extensions.ticket_training_v1_1.train import check_completed


def prerequisites():
    s.delivery_gate()
    check_completed("train")
    s.require(not (s.RUN / "launch.lock").exists(), "Original launcher still running/locked; do not remove its lock")
    s.require(not (s.RUN / "eval/dev").exists(), "Unexpected original DEV outputs; preserve them and inspect first")
    for variant in ("hermes-r8", "ticket-sft-v3"):
        run = s.read(s.eval_directory("smoke", variant) / "run.json")
        s.require(run["status"] == "complete" and run["stage"] == "smoke"
                  and run["variant"] == variant and len(run["case_ids"]) == 12,
                  "Both completed original 12-row smoke runs are required")
    failure = s.read(s.RUN / "logs/smoke-comparison.status.json")
    text = (s.RUN / "logs/smoke-comparison.log").read_text(encoding="utf-8")
    s.require(failure["status"] == "failed" and "ModuleNotFoundError: No module named 'validate'" in text,
              "Only the identified comparison-import failure is repaired by this entry point")


def step(arguments, key):
    log = s.EVAL / "logs" / (key + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    s.fresh(log)
    status = log.with_suffix(".status.json")
    s.fresh(status)
    command = [sys.executable, "-u", "-B", "-m", "extensions.ticket_training_v1_2.evaluate", *arguments]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               FORCE_TORCHRUN="0", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    record = {"status": "started", "started_at_utc": s.now(), "command": command,
              "allocated_gpu": "1", "repair_sources": s.local_sources()}
    s.write(status, record)
    try:
        with log.open("x", encoding="utf-8") as stream:
            print("Launching:", command, flush=True)
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    stream.write(line)
                    stream.flush()
                code = process.wait()
            except BaseException:
                process.terminate()  # Only this launcher's own child.
                process.wait()
                raise
        s.require(code == 0, f"Evaluation failed (exit={code}); keep outputs and log: {log}")
        record.update(status="complete", finished_at_utc=s.now(), exit_code=code, log_sha256=s.sha(log))
        s.update(status, record)
    except BaseException as exc:
        record.update(status="failed", finished_at_utc=s.now(), error=f"{type(exc).__name__}: {exc}")
        s.update(status, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("continue", "smoke-compare", "dev"), required=True)
    stage = parser.parse_args().stage
    os.chdir(ROOT)
    s.require(sys.version_info[:2] == (3, 11), "Use pinned remote Python 3.11")
    s.require(os.environ.get("WORLD_SIZE", "1") == "1" and os.environ.get("LOCAL_RANK", "-1") == "-1", "Single process only")
    prerequisites()
    if stage in ("continue", "smoke-compare"):
        s.fresh(s.EVAL)
        s.EVAL.mkdir(parents=True)
    else:
        s.require((s.EVAL / "smoke/comparison.json").is_file(), "Recompare smoke first")
    lock = s.EVAL / "launch.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream: stream.write(str(os.getpid()))
    except FileExistsError:
        raise ValueError("Evaluation lock exists; inspect it, do not auto-remove")
    try:
        if stage in ("continue", "smoke-compare"):
            step(["--stage", "smoke", "--compare"], "smoke-comparison")
        if stage in ("continue", "dev"):
            for variant in ("hermes-r8", "ticket-sft-v3"):
                step(["--stage", "dev", "--variant", variant], f"dev-{variant}")
            step(["--stage", "dev", "--compare"], "dev-comparison")
        print("Evaluation-only stages complete. No training rerun; semantics remain pending.", flush=True)
    finally:
        lock.unlink()


if __name__ == "__main__": main()
