"""Read-only release verification; optional fresh delivery package, never training."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"extensions/ticket_sft_v3_stage2"))
from pipeline import (approvals, check, dataset_info, digest, file_hash, load_bundle,
                      read_json, source_hashes, write_new)
from core import QUOTAS


def verify(directory, candidate, token_report):
    cm, episodes = load_bundle(candidate)
    reviews = approvals(candidate, cm)
    manifest = read_json(directory/"manifest.json")
    check(manifest["ready_for_training"] is True and manifest["rows"] == 3000,
          "Not a complete released dataset")
    check(manifest["revision"] == cm["revision"] and manifest["candidate_manifest_sha256"] == file_hash(candidate/"manifest.json")
          and manifest["pilot_gate"] == cm["gate"] and manifest["frozen_dev_sha256"] == cm["frozen_dev_sha256"],
          "Release provenance differs")
    check(manifest["source_sha256"] == source_hashes(), "Release sources changed")
    names = {"train.json", "dataset_info.json", "validation.json", "token-audit.json", "human-reviews.json"}
    check(set(manifest["files_sha256"]) == names, "Incomplete release file list")
    for name, sha in manifest["files_sha256"].items():
        check(file_hash(directory/name) == sha, f"Released file differs: {name}")
    train = read_json(directory/"train.json")
    check(train == [e["row"] for e in episodes], "Released training rows differ from verified episodes")
    check(len(train) == len({digest(r) for r in train}) == 3000, "Missing/duplicate training rows")
    check(Counter(e["spec"]["category"] for e in episodes) == Counter(QUOTAS), "Released quota mismatch")
    check(Counter(e["locale"] for e in episodes) == {"zh": 1500, "en": 1500}
          and manifest["locales"] == {"zh": 1500, "en": 1500}, "Released language mismatch")
    check(all(set(r) == {"system", "tools", "conversations"} for r in train), "Private metadata in training input")
    check(read_json(directory/"dataset_info.json") == dataset_info(), "Dataset registration differs")
    validation = read_json(directory/"validation.json")
    check(validation == read_json(candidate/"validation.json") and validation["passed"] is True
          and validation["failures"] == [] and len(validation["records"]) == 3000, "Validation report differs")
    human = read_json(directory/"human-reviews.json")
    actual = [read_json(p) for p in sorted((candidate/"reviews").glob("*.json"))]
    check(human["attestations"] == actual and human["hashes"] == reviews and len(actual) == 96,
          "Human decision evidence differs")
    check(manifest["human_review"] == "sampled_human_attestation" and manifest["human_review_pairs"] == 96,
          "Not a 96-pair human-attested release")
    report = read_json(directory/"token-audit.json")
    check(report == read_json(token_report) and manifest["token_report_sha256"] == file_hash(token_report),
          "Token report differs from uploaded original")
    check(report["passed"] is True and report["rows"] == report["all_turns_preserved"] == 3000
          and report["manifest_sha256"] == file_hash(candidate/"manifest.json")
          and report["source_sha256"] == source_hashes(), "Audit bound to another dataset/source")
    policy = {"template": "qwen", "cutoff_len": 4096, "train_on_prompt": False,
              "mask_history": False, "packing": False}
    check(manifest["training_policy"] == policy and all(report[k] == v for k, v in policy.items()),
          "Supervision policy differs")
    check(all(report[k] is False for k in ("train_on_prompt", "mask_history", "packing")), "Invalid policy types")
    check(report["versions"] == {"llamafactory": "0.9.3", "transformers": "4.51.3", "torch": "2.6.0+cu124"}
          and report["python"].startswith("3.11."), "Runtime stack differs")
    check(len(report["records"]) == len({r["id"] for r in report["records"]}) == 3000, "Audit records incomplete/duplicate")
    rounds = 0
    for record, episode in zip(report["records"], episodes):
        count = sum(m["from"] in {"gpt", "function_call"} for m in episode["row"]["conversations"])
        check(record["id"] == episode["id"] and record["passed"] is True
              and record["assistant_turns"] == len(record["target_spans"]) == count,
              "Audit episode or assistant turns differ")
        check(type(record["total_tokens"]) is int and 0 < record["total_tokens"] <= 4096, "Invalid episode length")
        previous_end, total = 0, 0
        for span in record["target_spans"]:
            check(all(type(span[k]) is int for k in ("start", "end", "target_tokens")), "Invalid span type")
            check(previous_end < span["start"] < span["end"] <= record["total_tokens"]
                  and span["end"]-span["start"] == span["target_tokens"], "Invalid supervision span")
            previous_end, total = span["end"], total+span["target_tokens"]
        check(previous_end == record["total_tokens"] and total == record["supervised_tokens"], "Incomplete supervision coverage")
        check(all(re.fullmatch(r"[0-9a-f]{64}", record[k]) for k in ("input_ids_sha256", "labels_sha256")),
              "Missing token/label hashes")
        rounds += count
    maximum = max(r["total_tokens"] for r in report["records"])
    check(report["max_tokens"] == maximum, "Maximum length differs")
    return {"passed": True, "rows": 3000, "pairs": 1500, "locales": manifest["locales"],
            "quotas": QUOTAS, "human_review_pairs": 96,
            "review_provenance": "Explicit current-chat user attestation; not an independently verified identity or signature",
            "token_audit": "remote report verified against candidate and all records, not a local tokenizer rerun",
            "assistant_turns": rounds, "max_tokens": maximum,
            "release_manifest_sha256": file_hash(directory/"manifest.json"),
            "token_report_sha256": file_hash(token_report), "ready_for_training": True, "training_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"data/ticket_sft_v3_stage2_train_release")
    parser.add_argument("--candidate", type=Path, default=ROOT/"data/ticket_sft_v3_candidate_stage2")
    parser.add_argument("--token-report", type=Path, default=ROOT/"reports/ticket-sft-v3-stage2-token-audit.json")
    parser.add_argument("--out", type=Path, help="Optional fresh ZIP path")
    parser.add_argument("--inventory", type=Path, help="Required with --out")
    args = parser.parse_args()
    summary = verify(args.data, args.candidate, args.token_report)
    if args.out:
        check(args.inventory is not None and not args.out.exists() and not args.inventory.exists(),
              "Use fresh package and inventory paths")
        files = sorted(set(args.data.glob("*.json")) | set((args.candidate/"reviews").glob("*.json"))
                       | {args.token_report, Path(__file__).resolve(),
                          ROOT/"tests/test_stage2_release_verifier.py",
                          ROOT/"reports/ticket-sft-v3-stage2-release-handoff-20261006.md"})
        members = {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in files}
        with zipfile.ZipFile(args.out, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for p in files:
                archive.write(p, p.relative_to(ROOT).as_posix())
        with zipfile.ZipFile(args.out) as archive:
            check(archive.testzip() is None, "Archive CRC failure")
            for name, metadata in members.items():
                check(hashlib.sha256(archive.read(name)).hexdigest() == metadata["sha256"], "Archive member hash failure")
            with tempfile.TemporaryDirectory(prefix="ticket-release-extract-") as temp:
                extracted = Path(temp)/"project"
                for name in ("ticket_agent_v1", "ticket_agent_v2"):
                    shutil.copytree(ROOT/"extensions"/name, extracted/"extensions"/name,
                                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                prior = ROOT.parent/"lora_sft_ticket_sft_v3_stage2_candidate.zip"
                with zipfile.ZipFile(prior) as original:
                    original.extractall(extracted)
                archive.extractall(extracted)
                run = subprocess.run([sys.executable, "-B", "scripts/verify_ticket_sft_stage2_release.py"],
                                     cwd=extracted, check=True, capture_output=True, text=True, encoding="utf-8")
                import json
                check(json.loads(run.stdout)["ready_for_training"] is True, "Extracted release verification failed")
                run = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p",
                                      "test_stage2_release_verifier.py", "-v"], cwd=extracted, check=True,
                                     capture_output=True, text=True, encoding="utf-8")
                check("Ran 7 tests" in run.stderr and "OK" in run.stderr, "Extracted release tests failed")
        summary.update({"archive": args.out.name, "archive_bytes": args.out.stat().st_size,
                        "archive_sha256": file_hash(args.out), "archive_crc_and_member_hashes": "passed",
                        "extracted_release_verification": "passed", "extracted_release_tests": 7,
                        "required_existing_files": "Stage2 candidate and extension from the previous delivery"})
        write_new(args.inventory, {**summary, "members": members})
    import json
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
