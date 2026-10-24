"""Offline CPU child launcher; remote/compare never starts a model or trainer."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.ticket_policy_audit_v1 import core as c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("local", "remote", "compare"), required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--model-path", type=Path, default=Path(c.s.MODEL))
    args = parser.parse_args()
    c.s.fresh(c.OUT / args.stage)
    logs = c.OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / (args.stage + ".log")
    c.s.fresh(log)
    env, removed = c.cpu_environment(os.environ)
    c.cpu_gate(env)
    command = [sys.executable, "-u", "-B", "-m", "extensions.ticket_policy_audit_v1.run", "--stage", args.stage,
               "--model-path", str(args.model_path)]
    if args.inputs:
        command += ["--inputs", str(args.inputs.resolve())]
    c.s.write(logs / (args.stage + "-environment.json"), {"removed_distributed_keys": removed,
        "CUDA_VISIBLE_DEVICES": "", "offline": True, "model_weights_loaded": False})
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        code = process.wait()
    c.s.require(code == 0, f"Audit failed (exit={code}); preserve output and log: {log}")


if __name__ == "__main__": main()
