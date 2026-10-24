"""Evaluation overlay. Never modify the hash-bound v1.1 training implementation."""
from extensions.ticket_training_v1_1.support import *
from extensions.ticket_training_v1_1 import support as predecessor

HERE = Path(__file__).resolve().parent
EVAL = RUN / "eval-v1.2"


def local_sources():
    files = sorted(HERE.rglob("*.py")) + [ROOT / "scripts/run_ticket_sft_v3_eval_v1_2.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def delivery_gate():
    predecessor.delivery_gate()
    inventory = read(ROOT / "reports/ticket-eval-v1.2-delivery.json")
    require(inventory["release_manifest_sha256"] == RELEASE_SHA and inventory["gpu"] == "1"
            and inventory["predecessor_delivery_sha256"] == sha(ROOT / "reports/ticket-training-v1.1-delivery.json"),
            "Wrong evaluation repair or changed predecessor")
    for name, expected in inventory["files_sha256"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe delivery path")
        require(sha(ROOT / relative) == expected, f"Repair file differs: {name}")


def frozen_descriptor():
    # Generic validate/core names stay inside a fresh child with an explicit trusted cwd.
    # Does not depend on release_gate, sys.path additions or the parent's sys.modules cache.
    directory = ROOT / "extensions/ticket_sft_v3_stage2"
    code = "from validate import frozen_dev; import json; print(json.dumps(frozen_dev(), ensure_ascii=True))"
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=directory, check=True,
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    data = json.loads(result.stdout)
    require(digest(data) == DEV_SHA and len(data["tasks"]) == 60, "Frozen DEV descriptor/source changed")
    return data


def dev_gate():
    descriptor = frozen_descriptor()
    legacy()
    from prepare_dev import load_corpus
    from review_tasks import validate_approvals
    from tasks import build_tasks
    tasks, refs, _ = load_corpus(DEV)
    require(tasks == build_tasks() == descriptor["tasks"] and len(tasks) == 60, "Frozen DEV tasks changed")
    approvals = validate_approvals(DEV, read(DEV / "review-checklist.json"))
    return tasks, refs, {"corpus_manifest_sha256": sha(DEV / "manifest.json"), "frozen_tasks_sha256": DEV_SHA,
                          "approval_sha256": approvals, "legacy_source_sha256": legacy().source_hashes()}


def eval_directory(stage, variant=None):
    require(stage in ("smoke", "dev"), "Unknown evaluation stage")
    if variant is not None:
        require(variant in ("hermes-r8", "ticket-sft-v3"), "Unknown adapter")
    base = RUN / "eval/smoke" if stage == "smoke" else EVAL / "dev"
    return base / variant if variant else base


def expected_eval_sources(stage):
    require(stage in ("smoke", "dev"), "Unknown evaluation stage")
    # Attribute completed smoke to its actual original code, never relabel it as v1.2.
    return predecessor.local_sources() if stage == "smoke" else local_sources()
