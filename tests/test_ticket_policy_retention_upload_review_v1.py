"""CPU-only uploaded evidence gates; no training or model execution."""
import copy
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import review_ticket_policy_retention_upload_v1 as u
from scripts import record_ticket_policy_retention_ai_review_v1 as a


class UploadReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = {name: (u.OUT / 'evidence' / name).read_bytes() for name in u.EXPECTED}

    def archive(self, entries):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / 'evidence.tar.gz'
        with tarfile.open(path, 'w:gz') as tar:
            for name, data, kind in entries:
                info = tarfile.TarInfo(name)
                info.type = kind
                if kind == tarfile.REGTYPE:
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
                else:
                    info.linkname = '../../outside'
                    tar.addfile(info)
        return path

    def test_complete_whitelist(self):
        entries = [(u.PREFIX + n, d, tarfile.REGTYPE) for n, d in self.files.items()]
        self.assertEqual(u.read_archive(self.archive(entries)), self.files)

    def test_unsafe_members(self):
        for name, kind in [('/etc/file', tarfile.REGTYPE), (u.PREFIX + '../x', tarfile.REGTYPE),
                           (u.PREFIX + 'remote\\x', tarfile.REGTYPE), (u.PREFIX + 'remote/C:x', tarfile.REGTYPE),
                           (u.PREFIX + 'remote/report.json', tarfile.SYMTYPE),
                           (u.PREFIX + 'remote/report.json', tarfile.LNKTYPE),
                           (u.PREFIX + 'remote/report.json', tarfile.CHRTYPE),
                           (u.PREFIX + 'remote/run.py', tarfile.REGTYPE)]:
            with self.subTest(name=name, kind=kind), self.assertRaises(ValueError):
                u.read_archive(self.archive([(name, b'{}', kind)]))

    def test_missing_member(self):
        with self.assertRaises(ValueError):
            u.read_archive(self.archive([(u.PREFIX + 'remote/report.json', b'{}', tarfile.REGTYPE)]))

    def test_duplicate_member(self):
        e = (u.PREFIX + 'remote/report.json', b'{}', tarfile.REGTYPE)
        with self.assertRaises(ValueError):
            u.read_archive(self.archive([e, e]))

    def test_strict_json(self):
        for data in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                u.parse(data)

    def test_valid_completed_cpu_logs(self):
        u.stage_logs(self.files)

    def mutate_json(self, name, key, value):
        files = dict(self.files)
        obj = u.parse(files[name])
        obj[key] = value
        files[name] = json.dumps(obj).encode()
        return files

    def test_environment_guards(self):
        for key, value in [('CUDA_VISIBLE_DEVICES', '1'), ('python', '3.13.9'), ('offline', False),
                           ('model_weights_loaded', True), ('training_started', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                u.stage_logs(self.mutate_json('logs/remote-environment.json', key, value))

    def test_incomplete_status(self):
        with self.assertRaises(ValueError):
            u.stage_logs(self.mutate_json('remote/status.json', 'status', 'failed'))

    def test_remote_report_hash(self):
        with self.assertRaises(ValueError):
            u.stage_logs(self.mutate_json('remote/report.json', 'rows', 1799))

    def test_missing_audit_progress(self):
        files = dict(self.files)
        files['logs/remote.log'] = files['logs/remote.log'].replace(b'Actual CPU tokenizer/labels: 1800/1800', b'missing')
        with self.assertRaises(ValueError):
            u.stage_logs(files)

    def test_no_output_overwrite(self):
        with self.assertRaises(ValueError):
            u.verify(Path('not-needed.tar.gz'), u.OUT)

    def test_review_gate_valid(self):
        evidence, sample, records = a.gate(u.OUT)
        self.assertEqual(len(sample), 120)
        self.assertEqual(len(records), 1800)
        self.assertFalse(evidence['ready_for_training'])

    def test_fixed_notes_coverage(self):
        self.assertEqual(sum(map(len, a.NOTES.values())), 60)
        with patch.dict(a.NOTES, {'authority': {}}, clear=False), self.assertRaises(ValueError):
            a.gate(u.OUT)

    def test_review_gate_changed_source(self):
        real_sha = a.c.sha
        def altered(path):
            return '0' * 64 if Path(path) == Path(u.__file__) else real_sha(path)
        with patch.object(a.c, 'sha', side_effect=altered), self.assertRaises(ValueError):
            a.gate(u.OUT)

    def test_no_human_approval_from_ai(self):
        e, _, _ = a.gate(u.OUT)
        self.assertEqual(e['human_acceptance'], 'pending')
        self.assertFalse(e['per_row_human_inspection_attested'])


if __name__ == '__main__':
    unittest.main()
