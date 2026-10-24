"""Reuse isolated prior guards, add accepted duplicate-retention checks."""
import ast
import copy
import importlib.util
from pathlib import Path
from unittest.mock import patch
import unittest
from extensions.ticket_policy_retention_release_v1_1 import core as r

# Private test namespace: the original tests and module globals remain unchanged.
_spec = importlib.util.spec_from_file_location('_retention_release_guard_copy', r.ROOT / 'tests/test_ticket_policy_retention_release_v1.py')
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
_legacy.r = r


class RetentionReleaseTests(_legacy.RetentionReleaseTests):
    def test_existing_output_refused(self):
        with self.assertRaises(ValueError): r.build('保留原数据发布')

    def test_no_training_or_gpu_imports(self):
        for path in (r.HERE / 'core.py', r.ROOT / 'scripts/release_ticket_policy_retention_v1_1.py', r.ROOT / 'scripts/package_ticket_policy_retention_release_v1_1.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            self.assertFalse(any(n.startswith(('torch', 'transformers', 'peft', 'llamafactory.chat', 'llamafactory.train')) for n in imports))

    def test_acceptance_keeps_ai_provenance(self):
        _, _, _, ids, binding = self.gate
        accepted = r.acceptance(ids, binding)
        self.assertEqual(accepted['user_message'], '保留原数据发布')
        self.assertEqual(accepted['prior_user_message'], '接受审核结论')
        self.assertEqual(len(accepted['pair_ids']), 60)
        self.assertTrue(accepted['AI_review_remains_AI_assisted'])
        self.assertFalse(accepted['per_row_human_inspection_attested'])
        self.assertFalse(accepted['training_authorized'])
        self.assertEqual(accepted['duplicate_handling'], 'keep_original_rows_without_deduplication')

    def test_duplicate_metrics_fixed(self):
        actual = r.check_duplicates(self.gate[1])
        self.assertEqual(actual, {'unique_public_train_rows': 1762, 'duplicate_public_row_groups': 38,
                                'excess_public_rows': 38, 'max_public_row_multiplicity': 2})

    def test_wrong_duplicate_metrics_manifest(self):
        self.changed_manifest(lambda m: m['duplicate_metrics'].update(unique_public_train_rows=1800))
        self.changed_manifest(lambda m: m['duplicate_metrics'].update(excess_public_rows=0))

    def test_wrong_duplicate_policy_manifest(self):
        self.changed_manifest(lambda m: m.update(duplicate_handling='deduplicated'))
        self.changed_manifest(lambda m: m.update(duplicate_disclosure_sha256='0' * 64))

    def test_disclosure_hash_wrong(self):
        original = r.c.sha
        with patch.object(r.c, 'sha', side_effect=lambda path: '0' * 64 if Path(path) == r.DISCLOSURE else original(path)), self.assertRaises(ValueError):
            r.check_duplicates(self.gate[1])

    def test_disclosure_group_membership_wrong(self):
        original = r.c.read
        altered = copy.deepcopy(original(r.DISCLOSURE))
        altered['groups'][0]['episode_ids'][0] = 'different-id'
        with patch.object(r.c, 'read', side_effect=lambda path: altered if Path(path) == r.DISCLOSURE else original(path)), self.assertRaises(ValueError):
            r.check_duplicates(self.gate[1])

    def test_old_failed_release_still_blocked(self):
        self.assertTrue((r.previous.DATA / 'failure.json').exists())
        with self.assertRaises(ValueError): r.previous.validate_release(r.previous.DATA, self.gate)


if __name__ == '__main__': unittest.main()
