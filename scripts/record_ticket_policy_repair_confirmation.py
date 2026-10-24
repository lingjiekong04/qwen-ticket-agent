"""Append design acceptance with hashes; do not mutate pilot or generate data."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/ticket-policy-repair-next-stage"
sys.path.insert(0, str(ROOT))
from extensions.ticket_policy_repair_pilot_v1 import core as c
from extensions.ticket_policy_repair_pilot_v1.pipeline import verify
s = c.s


def proposed_plan():
    return {
        "status": "proposed_pending_user_confirmation", "version": "ticket-policy-repair-build-v1", "seed": 2026100708,
        "unit": "full_episode", "rows": 1200, "bilingual_pairs": 600,
        "categories": {
            "authority": {"rows": 400, "pairs": 200, "counterfactual_groups": 100,
                "groups_by_variant": {"plain_create": 40, "override_create": 35, "assign": 25},
                "profiles_by_group": {"json": 34, "sentence": 33, "explicit": 33}},
            "clarify": {"rows": 300, "pairs": 150,
                "pairs_by_variant": {"email_stop": 40, "email_followup": 40, "title_followup": 40, "permission_before_missing": 30},
                "profiles_by_pair": {"json": 50, "sentence": 50, "explicit": 50}},
            "faithful": {"rows": 300, "pairs": 150,
                "pairs_by_variant": {"unassigned": 30, "partial": 30, "requester_not_found": 30, "ticket_not_found": 30, "empty_kb": 30},
                "profiles_by_pair": {"json": 50, "sentence": 50, "explicit": 50}},
            "priority": {"rows": 200, "pairs": 100,
                "pairs_by_variant": {"order": 40, "examples": 40, "no_sla": 20},
                "profiles_by_pair": {"json": 34, "sentence": 33, "explicit": 33}}},
        "human_sample_pairs": {"authority": 12, "clarify": 12, "faithful": 12, "priority": 12},
        "human_sample_rows": 96, "authority_sample_whole_groups": 6,
        "min_answer_frameworks_per_locale_authority_priority": 8, "max_exact_answer_share_per_category_locale": 0.2,
        "train_on_prompt": False, "mask_history": False, "packing": False, "cutoff_len": 4096,
        "old_data_mixed": False, "pilot_copied_to_training": False, "frozen_dev_train_rows": 0,
        "old_final_test_read": False, "bulk_generation_authorized": False, "bulk_generated": False,
        "real_tokenizer_audited": False, "ready_for_training": False, "training_authorized": False, "training_started": False}


def validate_plan(plan):
    s.require(sum(v["rows"] for v in plan["categories"].values()) == plan["rows"] == 1200, "Row quota mismatch")
    s.require(sum(v["pairs"] for v in plan["categories"].values()) == plan["bilingual_pairs"] == 600, "Pair quota mismatch")
    for name, category in plan["categories"].items():
        s.require(category["rows"] == category["pairs"] * 2, "Bilingual count mismatch")
        if name == "authority":
            groups = category["counterfactual_groups"]
            s.require(sum(category["groups_by_variant"].values()) == sum(category["profiles_by_group"].values()) == groups
                      and groups * 2 == category["pairs"], "Counterfactual count mismatch")
        else:
            s.require(sum(category["pairs_by_variant"].values()) == sum(category["profiles_by_pair"].values()) == category["pairs"], "Variant/profile quota mismatch")
    s.require(set(plan["human_sample_pairs"]) == set(plan["categories"])
              and sum(plan["human_sample_pairs"].values()) * 2 == plan["human_sample_rows"] == 96
              and plan["authority_sample_whole_groups"] * 2 == plan["human_sample_pairs"]["authority"], "Sampling quota mismatch")
    s.require(all(0 < plan["human_sample_pairs"][name] <= cat["pairs"] for name, cat in plan["categories"].items()), "Invalid sample count")
    for key in ("old_data_mixed", "pilot_copied_to_training", "old_final_test_read", "bulk_generation_authorized", "bulk_generated",
                "real_tokenizer_audited", "ready_for_training", "training_authorized", "training_started", "train_on_prompt", "mask_history", "packing"):
        s.require(plan[key] is False, "Scope/supervision changed: " + key)
    s.require(plan["frozen_dev_train_rows"] == 0 and plan["cutoff_len"] == 4096
              and plan["status"] == "proposed_pending_user_confirmation", "Isolation/config/status changed")


def main():
    for name in ("user-design-confirmation.json", "plan.json", "verification.json"):
        s.fresh(OUT / name)
    verify()
    pilot_report = ROOT / "reports/ticket-policy-repair-pilot-v1"
    delivery = s.read(pilot_report / "delivery.json")
    for name, expected in delivery["files_sha256"].items():
        s.require(s.sha(ROOT / name) == expected, "Confirmed pilot changed: " + name)
    archive_sha = s.sha(ROOT / "lora_sft_ticket_policy_repair_pilot_v1.zip")
    s.require(archive_sha == s.read(pilot_report / "archive-verification.json")["archive_sha256"], "ZIP changed")
    binding = {p.relative_to(ROOT).as_posix(): s.sha(p) for p in [c.OUTPUT / "manifest.json", c.OUTPUT / "episodes.json",
        c.OUTPUT / "validation.json", pilot_report / "ai-review.json", pilot_report / "delivery.json"]}
    plan = proposed_plan()
    validate_plan(plan)
    plan["plan_document_sha256"] = s.sha(OUT / "plan.md")
    plan["confirmed_pilot_manifest_sha256"] = s.sha(c.OUTPUT / "manifest.json")
    s.write(OUT / "user-design-confirmation.json", {
        "date": "2026-10-07", "design_confirmation": "accepted", "user_message": "接受", "follow_up_message": "继续",
        "scope": "18 bilingual representative pairs; acceptance of design only",
        "provenance": "Explicit current-chat user statement; no independently verified identity or signature",
        "rows": 36, "pairs": 18, "bound_files_sha256": binding, "pilot_zip_sha256": archive_sha,
        "ai_review_remains_ai_assisted": True, "per_row_human_inspection_attested": False,
        "batch_plan_approved": False, "training_authorized": False,
        "original_artifacts_modified": False, "ready_for_training": False})
    s.write(OUT / "plan.json", plan)
    s.write(OUT / "verification.json", {
        "passed": True, "design_confirmation_recorded": True, "pilot_hashes_verified": True,
        "pilot_rows_replayed": 36, "proposed_plan_quota_check": "passed", "proposed_rows": 1200, "proposed_pairs": 600,
        "files_sha256": {name: s.sha(OUT / name) for name in ("user-design-confirmation.json", "plan.json", "plan.md")},
        "recorder_source_sha256": s.sha(Path(__file__).resolve()), "bulk_generated": False, "training_started": False})
    print(json.dumps({"design_confirmation": "accepted", "proposed_rows": 1200, "proposed_pairs": 600,
                      "plan_status": plan["status"], "bulk_generated": False, "training_started": False}))


if __name__ == "__main__":
    main()
