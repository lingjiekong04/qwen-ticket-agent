#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extensions.ticket_runtime_model_recovery_v1 import support as model_support
from extensions.ticket_runtime_prospective_blind_v6_eval import support as s


def tests() -> dict:
    names = ("extensions.ticket_runtime_trusted_completion_v1_7.tests",
             "extensions.ticket_runtime_prospective_holdout_v6.tests",
             "extensions.ticket_runtime_prospective_blind_v6_eval.tests")
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors)}


def run_authorized() -> dict:
    s.verify(server=True)
    s.RUN.mkdir(parents=True)
    for arm in s.ARMS:
        (s.RUN / arm).mkdir()
    s.save_new(s.RUN / "authorization.json", {
        "user_message": s.AUTH_TEXT, "physical_gpu": "0", "planned_model_episodes": 320,
        "single_shot": True, "training_authorized": False,
        "live_tools_authorized": False, "deployment_authorized": False})
    gpu_env, removed = model_support.clean_environment(dict(os.environ), gpu=True)
    worker_log = s.RUN / "worker.log"
    command = [sys.executable, "-u", "-B", "-m",
               "extensions.ticket_runtime_prospective_blind_v6_eval.worker"]
    with worker_log.open("x", encoding="utf-8", newline="\n") as stream:
        code = subprocess.run(command, cwd=ROOT, env=gpu_env, stdout=stream,
                              stderr=subprocess.STDOUT).returncode
    s.require(code == 0, "v6 GPU worker failed; preserve output: " + str(worker_log))
    cpu_env, _ = model_support.clean_environment(dict(os.environ), gpu=False)
    compare_log = s.RUN / "compare.log"
    command = [sys.executable, "-u", "-B", "-m",
               "extensions.ticket_runtime_prospective_blind_v6_eval.compare"]
    with compare_log.open("x", encoding="utf-8", newline="\n") as stream:
        code = subprocess.run(command, cwd=ROOT, env=cpu_env, stdout=stream,
                              stderr=subprocess.STDOUT).returncode
    s.require(code == 0, "v6 CPU scoring failed; preserve output: " + str(compare_log))
    comparison = s.read(s.RUN / "comparison.json")
    complete = {
        "passed": True, "stage": "one_shot_independent_final_holdout_v6_complete",
        "episodes": 320, "physical_gpu": "0",
        "comparison_sha256": s.sha(s.RUN / "comparison.json"),
        "removed_distributed_environment_keys": removed,
        "formal_blind_acceptance": comparison["formal_blind_acceptance"],
        "usable_candidate_level": comparison["usable_candidate_level"],
        "holdout_acceptance_consumed": True, "semantic_review": "pending",
        "training_started": False, "live_tools_executed": 0, "deployment_authorized": False}
    s.save_new(s.RUN / "complete.json", complete)
    return {**complete, "comparison": comparison}


def verify_output() -> dict:
    complete, comparison = s.read(s.RUN / "complete.json"), s.read(s.RUN / "comparison.json")
    s.require(complete["episodes"] == comparison["episodes"] == 320 and
              complete["comparison_sha256"] == s.sha(s.RUN / "comparison.json") and
              comparison["holdout_acceptance_consumed"] is True,
              "v6 completion evidence changed")
    for arm in s.ARMS:
        rows = (s.RUN / arm / "episodes-unscored.jsonl").read_text(encoding="utf-8").splitlines()
        s.require(len(rows) == 160, "v6 arm is incomplete: " + arm)
    return {"passed": True, "stage": "CPU_one_shot_final_holdout_v6_output_verified",
            "episodes": 320, "comparison_sha256": complete["comparison_sha256"],
            "formal_blind_acceptance": comparison["formal_blind_acceptance"],
            "usable_candidate_level": comparison["usable_candidate_level"],
            "holdout_acceptance_consumed": True, "semantic_review": "pending",
            "training_started": False, "live_tools_executed": 0, "deployment_authorized": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("verify", "run", "verify-output"), required=True)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--authorize-gpu", action="store_true")
    args = parser.parse_args()
    if args.stage == "verify":
        checked = tests() if args.tests else None
        if checked:
            s.require(checked["passed"], "CPU tests failed")
        result = s.verify(server=args.server)
        if checked:
            result["tests"] = checked
    elif args.stage == "run":
        s.require(args.server and args.authorize_gpu,
                  "v6 GPU execution requires --server and explicit --authorize-gpu")
        result = run_authorized()
    else:
        s.require(not args.authorize_gpu, "verify-output never accepts GPU authorization")
        result = verify_output()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
