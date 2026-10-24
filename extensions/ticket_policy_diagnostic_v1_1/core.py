"""Single-process inference environment repair; original v1 stays hash-bound."""
import importlib.util
from pathlib import Path
from extensions.ticket_policy_diagnostic_v1.core import *
from extensions.ticket_policy_diagnostic_v1 import core as predecessor

HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "runs/ticket-policy-diagnostic-v1.1"
LOGS = ROOT / "runs/ticket-policy-diagnostic-v1.1-logs"
DIST_KEYS = frozenset(("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
    "ROLE_WORLD_SIZE", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT", "ACCELERATE_USE_DEEPSPEED",
    "ACCELERATE_USE_FSDP", "ACCELERATE_USE_TP", "ACCELERATE_USE_CPU"))
DIST_PREFIXES = ("TORCHELASTIC_", "OMPI_COMM_WORLD_", "PMI_", "PMIX_", "MV2_COMM_WORLD_")


def distributed_keys(environment):
    return sorted(key for key in environment if key in DIST_KEYS or key.startswith(DIST_PREFIXES))


def child_environment(parent):
    # WORLD_SIZE=1 is NOT harmless: transformers 4.51.3 activates tp_plan
    # with device_map='auto' for any nonzero WORLD_SIZE. LOCAL_RANK=-1 can
    # also produce cuda:-1 in framework helpers. Remove, never fabricate ranks.
    removed = distributed_keys(parent)
    clean = {key: value for key, value in parent.items() if key not in removed}
    clean.update(CUDA_VISIBLE_DEVICES="1", FORCE_TORCHRUN="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                 PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    return clean, removed


def environment_gate(environment):
    s.require(not distributed_keys(environment), "Distributed launcher variables are present; use the v1.1 launcher to sanitize its child environment")
    s.require(environment.get("CUDA_VISIBLE_DEVICES") == "1" and environment.get("FORCE_TORCHRUN", "0") == "0"
              and environment.get("HF_HUB_OFFLINE") == environment.get("TRANSFORMERS_OFFLINE") == "1",
              "Require GPU1, offline, non-torchrun execution")


def sources():
    files = sorted(HERE.rglob("*.py")) + [ROOT / "scripts/run_ticket_policy_diagnostic_v1_1.py"]
    return {path.relative_to(ROOT).as_posix(): s.sha(path) for path in files}


def delivery_gate():
    predecessor.delivery_gate()
    manifest = s.read(ROOT / "reports/ticket-policy-diagnostic-v1.1-delivery.json")
    s.require(manifest["predecessor_sha256"] == s.sha(ROOT / "reports/ticket-policy-diagnostic-v1-delivery.json"),
              "Wrong diagnostic repair predecessor")
    for name, expected in manifest["files_sha256"].items():
        relative = Path(name)
        s.require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe repair delivery path")
        s.require(s.sha(ROOT / relative) == expected, f"Repair file changed: {name}")


def isolated_runner():
    # A private module instance executes the unmodified v1 implementation.
    # Only its dependency pointer changes to this repair namespace. Original
    # module globals, manifests, task IDs/prompts and failed outputs stay intact.
    name = "extensions.ticket_policy_diagnostic_v1._private_runner_v1_1"
    spec = importlib.util.spec_from_file_location(name, ROOT / "extensions/ticket_policy_diagnostic_v1/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import sys
    module.c = sys.modules[__name__]
    module.s = s
    return module
