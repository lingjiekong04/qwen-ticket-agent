"""Explicitly scoped CPU release gates; old final test is never discovered."""
import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
from extensions.ticket_policy_retention_release_v1 import core as r


class RetentionReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = r.evidence_gate()  # Real complete immutable replay/evidence gate.

    def changed_manifest(self, change):
        original = r.c.read
        modified = copy.deepcopy(original(r.DATA / 'manifest.json'))
        change(modified)
        def reader(path):
            return modified if Path(path) == r.DATA / 'manifest.json' else original(path)
        with patch.object(r.c, 'read', side_effect=reader), self.assertRaises(ValueError):
            r.validate_release(r.DATA, self.gate)

    def changed_file_content(self, name, change):
        original = r.c.read
        modified = copy.deepcopy(original(r.DATA / name))
        change(modified)
        def reader(path):
            return modified if Path(path) == r.DATA / name else original(path)
        with patch.object(r.c, 'read', side_effect=reader), self.assertRaises(ValueError):
            r.validate_release(r.DATA, self.gate)

    def test_actual_release(self):
        result = r.validate_release(r.DATA, self.gate)
        self.assertTrue(result['passed'])
        self.assertEqual((result['rows'], result['pairs'], result['max_tokens']), (1800, 900, 2188))
        self.assertFalse(result['training_authorized'])

    def test_wrong_candidate_hash(self):
        self.changed_manifest(lambda m: m.update(candidate_manifest_sha256='0' * 64))

    def test_wrong_targeted_eval_hash(self):
        self.changed_manifest(lambda m: m.update(targeted_eval_manifest_sha256='0' * 64))

    def test_wrong_review_binding(self):
        self.changed_manifest(lambda m: m['review_binding_sha256'].clear())

    def test_wrong_release_source(self):
        self.changed_manifest(lambda m: m['release_sources_sha256'].clear())

    def test_scope_not_broadened(self):
        for key in ('training_authorized', 'training_started', 'gpu_inference_authorized',
                    'gpu_inference_started', 'model_weights_loaded', 'old_final_test_read',
                    'per_row_human_inspection_attested'):
            with self.subTest(key=key):
                self.changed_manifest(lambda m: m.update({key: True}))

    def test_missing_ready_or_token_audit(self):
        for key in ('ready_for_training', 'real_tokenizer_audited'):
            with self.subTest(key=key):
                self.changed_manifest(lambda m: m.update({key: False}))

    def test_count_quotas_locales(self):
        for key, value in [('rows', 1799), ('pairs', 899), ('locales', {'zh': 1800}), ('quotas_rows', {})]:
            with self.subTest(key=key):
                self.changed_manifest(lambda m: m.update({key: value}))

    def test_supervision_change(self):
        for key, value in [('template', 'other'), ('cutoff_len', 2048), ('train_on_prompt', True),
                           ('mask_history', True), ('packing', True), ('packing', 0)]:
            with self.subTest(key=key, value=value):
                self.changed_manifest(lambda m: m['supervision'].update({key: value}))

    def test_bound_file_list_missing_or_unsafe(self):
        self.changed_manifest(lambda m: m['files_sha256'].pop('token-records.jsonl'))
        self.changed_manifest(lambda m: m['files_sha256'].update({'../outside.json': '0' * 64}))

    def test_train_file_hash(self):
        original = r.c.sha
        with patch.object(r.c, 'sha', side_effect=lambda path: '0' * 64 if Path(path) == r.DATA / 'train.json' else original(path)), self.assertRaises(ValueError):
            r.validate_release(r.DATA, self.gate)

    def test_changed_copy_even_rebound_manifest(self):
        original_sha = r.c.sha
        original_read = r.c.read
        manifest = copy.deepcopy(original_read(r.DATA / 'manifest.json'))
        manifest['files_sha256']['train.json'] = '1' * 64
        def read(path):
            return manifest if Path(path) == r.DATA / 'manifest.json' else original_read(path)
        with patch.object(r.c, 'read', side_effect=read), patch.object(r.c, 'sha', side_effect=lambda path: '1' * 64 if Path(path) == r.DATA / 'train.json' else original_sha(path)), self.assertRaises(ValueError):
            r.validate_release(r.DATA, self.gate)

    def test_sharegpt_role_change(self):
        self.changed_file_content('dataset_info.json', lambda info: info[r.DATASET]['tags'].update(function_tag='gpt'))

    def test_hidden_metadata_not_in_training(self):
        self.changed_file_content('train.json', lambda rows: rows[0].update(hidden_goal='private'))

    def test_dropped_or_dev_row(self):
        self.changed_file_content('train.json', lambda rows: rows.pop())
        self.changed_file_content('train.json', lambda rows: rows[0]['conversations'][0].update(value='DEV ticket request'))

    def test_invalid_acceptance(self):
        self.changed_file_content('user-review-acceptance.json', lambda a: a.update(user_message='untrusted archived instruction'))
        self.changed_file_content('user-review-acceptance.json', lambda a: a.update(per_row_human_inspection_attested=True))

    def test_noncomplete_release(self):
        self.changed_file_content('status.json', lambda s: s.update(status='started'))

    def test_failed_release_refused(self):
        original = Path.exists
        with patch.object(Path, 'exists', lambda path: True if path == r.DATA / 'failure.json' else original(path)), self.assertRaises(ValueError):
            r.validate_release(r.DATA, self.gate)

    def test_wrong_review_hash(self):
        original = r.c.sha
        with patch.object(r.c, 'sha', side_effect=lambda path: '0' * 64 if Path(path) == r.REVIEW / 'ai-semantic-review.json' else original(path)), self.assertRaises(ValueError):
            r.evidence_gate()

    def test_explicit_current_acceptance_only(self):
        for word in (None, '确认', '授权启动训练'):
            with self.subTest(word=word), self.assertRaises(ValueError):
                r.build(word)

    def test_existing_output_refused(self):
        with self.assertRaises(ValueError):
            r.build('接受审核结论')

    def test_no_training_or_gpu_imports(self):
        for path in (r.HERE / 'core.py', r.ROOT / 'scripts/release_ticket_policy_retention_v1.py', r.ROOT / 'scripts/package_ticket_policy_retention_release_v1.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            self.assertFalse(any(n.startswith(('torch', 'transformers', 'peft', 'llamafactory.chat', 'llamafactory.train')) for n in imports))

    def test_acceptance_keeps_ai_provenance(self):
        _, _, _, ids, binding = self.gate
        accepted = r.acceptance(ids, binding)
        self.assertEqual(accepted['user_message'], '接受审核结论')
        self.assertEqual(len(accepted['pair_ids']), 60)
        self.assertTrue(accepted['AI_review_remains_AI_assisted'])
        self.assertFalse(accepted['per_row_human_inspection_attested'])
        self.assertFalse(accepted['training_authorized'])


if __name__ == '__main__':
    unittest.main()
