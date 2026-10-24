"""Package verified data and new evidence only. Never train or overwrite outputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.ticket_policy_repair_release_v1 import core as r
s = r.s


def main():
    archive_path = ROOT / 'lora_sft_ticket_policy_repair_train_release_v1.zip'
    inventory_path = r.REPORT / 'delivery.json'
    for path in (archive_path, inventory_path, r.REPORT / 'tests.log', r.REPORT / 'archive-verification.json'):
        s.fresh(path)
    summary = r.verify()
    s.require({p.name for p in (ROOT / 'tests').glob('test_policy_repair_*e*_v1.py')}
              == {'test_policy_repair_release_v1.py', 'test_policy_repair_uploaded_evidence_v1.py'}, 'Unexpected test discovery scope')
    result = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests',
        '-p', 'test_policy_repair_*e*_v1.py', '-v'], cwd=ROOT,
        env=dict(os.environ, PYTHONIOENCODING='utf-8'), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (r.REPORT / 'tests.log').write_bytes(result.stdout)
    # Explicitly scoped runs avoid accidentally discovering old/final-test files.
    s.require(result.returncode == 0 and b'Ran 42 tests' in result.stdout and b'\nOK' in result.stdout, 'Release/evidence tests failed; preserve log')
    members = set(r.DATA.glob('*.json')) | {r.DATA / 'token-records.jsonl'}
    members |= {ROOT / name for name in r.sources()}
    members |= {p for p in r.REVIEW.rglob('*') if p.is_file()}
    members |= {r.REPORT / name for name in ('user-review-acceptance.json', 'verification.json', 'status.json', 'tests.log', 'handoff.md')}
    # Immutable dependencies are not copied; the prior candidate package must already be installed.
    paths = sorted(members)
    s.require(all(p.is_file() and p.resolve().is_relative_to(ROOT.resolve()) for p in paths), 'Missing or external package file')
    metadata = {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size, 'sha256': s.sha(p)} for p in paths}
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for p in paths: archive.write(p, p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        s.require(archive.testzip() is None and set(archive.namelist()) == set(metadata), 'Archive CRC/member list failure')
        for name, value in metadata.items():
            s.require(hashlib.sha256(archive.read(name)).hexdigest() == value['sha256'], 'Archived bytes differ: ' + name)
        with tempfile.TemporaryDirectory(prefix='release-extracted-') as temp:
            extracted = Path(temp) / 'release'
            extracted.mkdir()
            for name in [*r.RELEASE_FILES, 'manifest.json', 'status.json']:
                (extracted / name).write_bytes(archive.read((r.DATA / name).relative_to(ROOT).as_posix()))
            extracted_result = r.verify(extracted)
            s.require(extracted_result == summary, 'Extracted data verification differs')
    inventory = {**summary, 'kind': 'incremental data release; prior candidate/CPU audit dependencies required',
        'archive': archive_path.name, 'archive_bytes': archive_path.stat().st_size,
        'archive_sha256': s.sha(archive_path), 'members': metadata, 'tests_passed': 42,
        'crc_and_all_member_hashes': 'passed',
        'extracted_release_data_verification': 'passed using unchanged local dependency tree; not a clean-machine installation'}
    s.write(inventory_path, inventory)
    s.write(r.REPORT / 'archive-verification.json', {k: v for k, v in inventory.items() if k != 'members'})
    print(json.dumps({k: v for k, v in inventory.items() if k != 'members'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
