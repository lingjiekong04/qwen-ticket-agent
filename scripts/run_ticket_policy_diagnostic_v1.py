"""GPU1/offline inference only. New outputs; no automatic resume or overwrite."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.ticket_policy_diagnostic_v1 import core as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "prepare", "hermes-r8", "ticket-sft-v3", "compare"), default="all")
    args = parser.parse_args()
    c.delivery_gate()
    if args.stage == "all": c.s.fresh(c.OUTPUT)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="1", WORLD_SIZE="1", LOCAL_RANK="-1", FORCE_TORCHRUN="0",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    stages = ("prepare", "hermes-r8", "ticket-sft-v3", "compare") if args.stage == "all" else (args.stage,)
    for stage in stages:
        command = [sys.executable, "-u", "-B", "-m", "extensions.ticket_policy_diagnostic_v1.run", "--stage", stage]
        print("Launching:", command, flush=True)
        # Capture each child including admission/loading failures. Failed outputs
        # and logs are retained; explicit later stages may run only into fresh dirs.
        logs = ROOT / "runs/ticket-policy-diagnostic-v1-logs"
        logs.mkdir(exist_ok=True)
        log = logs / f"{stage}.log"
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8")
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            code = process.wait()
        c.s.require(code == 0, f"Diagnostic failed (exit={code}); preserve outputs and log: {log}")
    print("Diagnostic stage finished. Training was not started; answer semantics remain pending review.")


if __name__ == "__main__": main()
