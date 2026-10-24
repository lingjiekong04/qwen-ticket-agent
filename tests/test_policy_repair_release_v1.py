"""Release mutation tests use disposable copies; accepted artifacts stay intact."""
import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from extensions.ticket_policy_repair_release_v1 import core as r


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proof = r.evidence_gate()  # Full gates/replay once, before cached mutation tests.

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='repair-release-test-')
        self.directory = Path(self.temp.name) / 'release'
        shutil.copytree(r.DATA, self.directory)
        self.gate = patch.object(r, 'evidence_gate', return_value=copy.deepcopy(self.proof))
        self.gate.start()

    def tearDown(self):
        self.gate.stop()
        self.temp.cleanup()

    def write(self, name, value):
        r.s.update(self.directory / name, value)

    def rebind_status(self):
        value = r.s.read(self.directory / 'status.json')
        value['manifest_sha256'] = r.s.sha(self.directory / 'manifest.json')
        self.write('status.json', value)

    def change_manifest(self, change):
        value = r.s.read(self.directory / 'manifest.json')
        change(value)
        self.write('manifest.json', value)
        self.rebind_status()
        with self.assertRaises(ValueError): r.verify(self.directory)

    def change_file(self, name, change, rebind=True):
        value = r.s.read(self.directory / name)
        change(value)
        self.write(name, value)
        if rebind:
            manifest = r.s.read(self.directory / 'manifest.json')
            manifest['files_sha256'][name] = r.s.sha(self.directory / name)
            self.write('manifest.json', manifest)
            self.rebind_status()
        with self.assertRaises(ValueError): r.verify(self.directory)

    def test_valid_release(self):
        summary = r.verify(self.directory)
        self.assertEqual(summary['rows'], 1200)
        self.assertTrue(summary['ready_for_training'])
        self.assertFalse(summary['training_authorized'])
        self.assertFalse(summary['per_row_human_inspection_attested'])

    def test_old_candidate_still_pending(self):
        value = r.s.read(r.c.OUT / 'manifest.json')
        self.assertFalse(value['ready_for_training'])
        self.assertEqual(value['human_review'], 'pending')

    def test_candidate_hash_wrong(self):
        self.change_manifest(lambda v: v.update(candidate_manifest_sha256='0' * 64))

    def test_old_source_binding_wrong(self):
        self.change_manifest(lambda v: v['candidate_sources'].clear())

    def test_release_source_wrong(self):
        self.change_manifest(lambda v: v['release_sources'].clear())

    def test_supervision_change(self):
        for key in ('train_on_prompt', 'mask_history', 'packing'):
            with self.subTest(key=key):
                original = (r.DATA / 'manifest.json').read_bytes()
                (self.directory / 'manifest.json').write_bytes(original)
                self.change_manifest(lambda v: v['training_policy'].update({key: True}))

    def test_false_instead_of_boolean(self):
        self.change_manifest(lambda v: v['training_policy'].update(train_on_prompt=0))

    def test_length_changed(self):
        self.change_manifest(lambda v: v['training_policy'].update(cutoff_len=2048))

    def test_training_authorization_inferred(self):
        self.change_manifest(lambda v: v.update(training_authorized=True))

    def test_training_started(self):
        self.change_manifest(lambda v: v.update(training_started=True))

    def test_quotas_wrong(self):
        self.change_manifest(lambda v: v['quotas'].update(authority=399))

    def test_locale_count_wrong(self):
        self.change_manifest(lambda v: v['locales'].update(zh=599))

    def test_missing_file_even_when_manifest_rebound(self):
        self.change_manifest(lambda v: v['files_sha256'].pop('token-records.jsonl'))

    def test_unsafe_file_list(self):
        self.change_manifest(lambda v: v['files_sha256'].update({'../escape': '0' * 64}))

    def test_train_bytes_changed(self):
        self.change_file('train.json', lambda v: v[0]['conversations'][-1].update(value='fake success'), False)

    def test_deleted_episode_with_hash_rebound(self):
        self.change_file('train.json', lambda v: v.pop())

    def test_hidden_metadata_with_hash_rebound(self):
        self.change_file('train.json', lambda v: v[0].update(fixture={}))

    def test_wrong_roles_with_hash_rebound(self):
        self.change_file('dataset_info.json', lambda v: v[r.DATASET]['tags'].update(observation_tag='gpt'))

    def test_fabricated_per_pair_human_inspection(self):
        self.change_file('user-review-acceptance.json', lambda v: v.update(per_row_human_inspection_attested=True))

    def test_missing_accepted_pair(self):
        self.change_file('user-review-acceptance.json', lambda v: v['pair_ids'].pop())

    def test_ai_is_not_human_approval(self):
        self.change_manifest(lambda v: v.update(human_review='all_rows_human_approved'))

    def test_incomplete_status(self):
        self.write('status.json', {'status': 'failed'})
        with self.assertRaises(ValueError): r.verify(self.directory)

    def test_existing_output_rejected(self):
        with self.assertRaises(ValueError): r.build('接受')

    def test_no_user_acceptance(self):
        with self.assertRaises(ValueError): r.build(None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
