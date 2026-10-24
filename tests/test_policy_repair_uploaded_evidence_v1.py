"""Regression tests for uploaded evidence gates; no tokenizer or model loading."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('upload_review', ROOT / 'scripts/review_ticket_policy_repair_upload_v1.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class UploadedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = ROOT / 'reports/ticket-policy-repair-build-v1/upload-review-20261007/evidence'
        cls.files = {name: (directory / name).read_bytes() for name in review.EXPECTED}
        cls.episodes = review.s.read(review.c.OUT / 'episodes.json')
        cls.records = [review.parse(line.encode()) for line in cls.files['remote/records.jsonl'].decode().splitlines()]

    def rejected_record(self, change):
        records = copy.deepcopy(self.records)
        change(records)
        with self.assertRaises(ValueError):
            review.check_records(records, self.episodes)

    def rejected_file(self, name, change, rebind=False):
        files = dict(self.files)
        value = review.parse(files[name])
        change(value)
        files[name] = review.c.audit.canonical(value).encode()
        if rebind:
            stage = name.split('/')[0]
            status_name = stage + '/status.json'
            status = review.parse(files[status_name])
            status['report_sha256'] = review.sha(files[name])
            files[status_name] = review.c.audit.canonical(status).encode()
        with self.assertRaises(ValueError):
            review.check_evidence(files, self.episodes)

    def test_valid_evidence(self):
        remote, records, sample = review.check_evidence(self.files, self.episodes)
        self.assertEqual((len(records), len(sample), remote['max_tokens']), (1200, 96, 1868))

    def test_missing_record(self):
        self.rejected_record(lambda r: r.pop())

    def test_duplicate_record(self):
        self.rejected_record(lambda r: r.__setitem__(1, copy.deepcopy(r[0])))

    def test_changed_row_hash(self):
        self.rejected_record(lambda r: r[0].update(row_sha256='0' * 64))

    def test_truncated_length(self):
        self.rejected_record(lambda r: r[0].update(total_tokens=4097))

    def test_missing_permission(self):
        self.rejected_record(lambda r: r[0]['permission_span'].update(text=''))

    def test_permission_after_target(self):
        self.rejected_record(lambda r: r[0]['permission_span'].update(end=r[0]['total_tokens']))

    def test_permission_tokens_changed(self):
        self.rejected_record(lambda r: r[0]['permission_span']['token_ids'].__setitem__(0, 1))

    def test_assistant_masked(self):
        self.rejected_record(lambda r: r[0].update(all_assistant_turns_supervised=False))

    def test_user_or_tool_supervised(self):
        self.rejected_record(lambda r: r[0].update(public_source_spans_masked=False))

    def test_target_accounting_changed(self):
        self.rejected_record(lambda r: r[0].update(text_target_tokens=r[0]['text_target_tokens'] + 1))

    def test_source_manifest_changed_even_with_status_rebound(self):
        self.rejected_file('remote/report.json', lambda r: r.update(candidate_manifest_sha256='0' * 64), True)

    def test_unapproved_release_flag(self):
        self.rejected_file('compare/report.json', lambda r: r.update(ready_for_training=True), True)

    def test_wrong_version(self):
        self.rejected_file('remote/report.json', lambda r: r['versions'].update(transformers='wrong'), True)

    def test_gpu_environment(self):
        self.rejected_file('logs/remote-environment.json', lambda r: r.update(CUDA_VISIBLE_DEVICES='1'))

    def test_review_sample_tampered(self):
        self.rejected_file('remote/review-pairs.json', lambda r: r.pop())

    def test_unsafe_or_missing_archive_members(self):
        for name in ('3/../escape.json', '/escape.json', '3/bad\\name.json', '3/C:escape.json', '3/unknown.json', '3/remote/report.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'bad.zip'
                with zipfile.ZipFile(path, 'w') as archive:
                    archive.writestr(name, '{}')
                with self.assertRaises(ValueError):
                    review.read_archive(path)

    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                review.s.fresh(Path(temp))


if __name__ == '__main__':
    unittest.main(verbosity=2)
