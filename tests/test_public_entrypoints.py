import copy
import tempfile
from pathlib import Path
import unittest
from scripts.run_public_demo import relocated_config

class PublicEntryTests(unittest.TestCase):
    def test_relocation_changes_only_asset_paths(self):
        original = {'model_path': '/old', 'adapter_path': '/old-adapter',
                    'stack': {'torch': '2.6.0+cu124'}, 'adapter_files_sha256': {'a': 'frozen'},
                    'backend': 'session_isolated_in_memory'}
        before = copy.deepcopy(original)
        with tempfile.TemporaryDirectory() as directory:
            changed = relocated_config(original, directory, directory)
        self.assertEqual(original, before)
        self.assertEqual({k:v for k,v in changed.items() if not k.endswith('_path')},
                         {k:v for k,v in original.items() if not k.endswith('_path')})

    def test_relocation_requires_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                relocated_config({}, Path(directory) / 'missing', directory)

if __name__ == '__main__':
    unittest.main()
