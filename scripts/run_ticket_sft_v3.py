"""Sequential fresh-stage launcher on GPU 1. No remote access or automatic resume."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.ticket_training_v1 import support as s


def step(module, arguments, key):
    log = s.RUN / "logs" / (key + ".log")
    s.fresh(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    status_path = log.with_suffix(".status.json")
    s.fresh(status_path)
    command = [sys.executable, "-u", "-B", "-m", "extensions.ticket_training_v1." + module, *arguments]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               FORCE_TORCHRUN="0", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    record = {"status": "started", "started_at_utc": s.now(), "command": command,
              "allocated_gpu": "1", "sources": s.local_sources()}
    s.write(status_path, record)
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
                # Only this launcher's own child, never another GPU user's process.
                process.terminate()
                process.wait()
                raise
        s.require(code == 0, f"Stage failed (exit={code}); keep log and outputs: {log}")
        record.update(status="complete", finished_at_utc=s.now(), exit_code=code, log_sha256=s.sha(log))
        s.update(status_path, record)
    except BaseException as exc:
        record.update(status="failed", finished_at_utc=s.now(), error=f"{type(exc).__name__}: {exc}")
        s.update(status_path, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "train", "evaluate", "all"), required=True)
    stage = parser.parse_args().stage
    os.chdir(ROOT)
    s.require(sys.version_info[:2] == (3, 11), "Use the remote pinned Python 3.11 executable")
    s.require(os.environ.get("WORLD_SIZE", "1") == "1" and os.environ.get("LOCAL_RANK", "-1") == "-1", "Single process only")
    s.delivery_gate()
    if stage in ("preflight", "all"):
        s.fresh(s.RUN)
    else:
        s.require(s.RUN.is_dir(), "Run preflight first")
    s.RUN.mkdir(parents=True, exist_ok=True)
    lock = s.RUN / "launch.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        raise ValueError("Launcher lock exists; inspect interrupted/running work, do not auto-remove")
    try:
        if stage in ("preflight", "all"):
            step("train", ["--stage", "preflight"], "preflight")
        if stage in ("train", "all"):
            step("train", ["--stage", "train"], "train")
        if stage in ("evaluate", "all"):
            for phase in ("smoke", "dev"):
                for variant in ("hermes-r8", "ticket-sft-v3"):
                    step("evaluate", ["--stage", phase, "--variant", variant], f"{phase}-{variant}")
                step("evaluate", ["--stage", phase, "--compare"], f"{phase}-comparison")
        print("Requested stages completed. DEV semantics remain pending review; no final-test access.", flush=True)
    finally:
        lock.unlink()  # Exact owned lock only; all failure logs/data are retained.


if __name__ == "__main__":
    main()
