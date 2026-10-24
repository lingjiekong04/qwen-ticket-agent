"""Tampering checks are isolated to disposable copies of the released dataset."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import verify_ticket_sft_stage2_release as verifier


class ReleaseVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = ROOT/"data/ticket_sft_v3_candidate_stage2"
        cls.original = ROOT/"data/ticket_sft_v3_stage2_train_release"
        cls.report = ROOT/"reports/ticket-sft-v3-stage2-token-audit.json"
        cls.bundle = verifier.load_bundle(cls.candidate)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ticket-release-test-")
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)/"release"
        shutil.copytree(self.original, self.data)
        self.token_report = Path(self.temp.name)/"token-report.json"
        shutil.copyfile(self.report, self.token_report)
        # Replay was independently done once above. Avoid repeating expensive replay
        # for each deliberately corrupted disposable export.
        self.replay = patch.object(verifier, "load_bundle", return_value=self.bundle)
        self.replay.start()
        self.addCleanup(self.replay.stop)

    def write_json(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    def refresh_file_hash(self, name):
        path = self.data/"manifest.json"
        manifest = verifier.read_json(path)
        manifest["files_sha256"][name] = verifier.file_hash(self.data/name)
        if name == "token-audit.json":
            manifest["token_report_sha256"] = verifier.file_hash(self.token_report)
        self.write_json(path, manifest)

    def mutate_token_record(self, change):
        value = verifier.read_json(self.data/"token-audit.json")
        change(value)
        self.write_json(self.data/"token-audit.json", value)
        self.write_json(self.token_report, value)
        self.refresh_file_hash("token-audit.json")

    def run_verify(self):
        return verifier.verify(self.data, self.candidate, self.token_report)

    def test_complete_release_passes(self):
        result = self.run_verify()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(result["assistant_turns"], 7758)
        self.assertFalse(result["training_started"])

    def test_modified_train_bytes_rejected(self):
        with (self.data/"train.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "Released file differs"):
            self.run_verify()

    def test_missing_human_record_rejected_even_with_refreshed_hash(self):
        value = verifier.read_json(self.data/"human-reviews.json")
        value["attestations"].pop()
        self.write_json(self.data/"human-reviews.json", value)
        self.refresh_file_hash("human-reviews.json")
        with self.assertRaisesRegex(ValueError, "Human decision evidence"):
            self.run_verify()

    def test_false_span_rejected_even_with_refreshed_hashes(self):
        self.mutate_token_record(lambda r: r["records"][0]["target_spans"][0].update(target_tokens=14))
        with self.assertRaisesRegex(ValueError, "Invalid supervision span"):
            self.run_verify()

    def test_lost_assistant_round_rejected(self):
        self.mutate_token_record(lambda r: r["records"][2].update(assistant_turns=3))
        with self.assertRaisesRegex(ValueError, "assistant turns differ"):
            self.run_verify()

    def test_wrong_prompt_mask_policy_rejected(self):
        self.mutate_token_record(lambda r: r.update(train_on_prompt=True))
        with self.assertRaisesRegex(ValueError, "Supervision policy"):
            self.run_verify()

    def test_other_candidate_manifest_rejected(self):
        value = verifier.read_json(self.data/"manifest.json")
        value["candidate_manifest_sha256"] = "0"*64
        self.write_json(self.data/"manifest.json", value)
        with self.assertRaisesRegex(ValueError, "Release provenance"):
            self.run_verify()


if __name__ == "__main__":
    unittest.main()
