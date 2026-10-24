"""Exclusive incremental data release package; no training/GPU execution."""
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
from extensions.ticket_policy_retention_release_v1_1 import core as r
from extensions.ticket_policy_audit_v1 import core as audit


def main():
    env, _ = audit.cpu_environment(os.environ)
    os.environ.clear(); os.environ.update(env)
    archive_path = ROOT / 'lora_sft_ticket_policy_retention_train_release_v1_1.zip'
    for path in (archive_path, r.REPORT / 'delivery.json', r.REPORT / 'tests.log', r.REPORT / 'archive-verification.json'):
        r.p.fresh(path)
    gate = r.evidence_gate()
    verified = r.validate_release(r.DATA, gate)
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest',
        'tests.test_ticket_policy_retention_release_v1_1', 'tests.test_ticket_policy_retention_upload_review_v1', '-v'],
        cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING='utf-8'), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (r.REPORT / 'tests.log').open('xb') as f: f.write(tests.stdout)
    r.c.require(tests.returncode == 0 and b'Ran 44 tests' in tests.stdout and b'\nOK' in tests.stdout, 'Release/evidence tests failed; log preserved')
    paths = set(r.DATA.glob('*.json')) | {r.DATA / 'token-records.jsonl'}
    paths |= {ROOT / name for name in r.sources()}
    paths |= {f for f in r.REVIEW.rglob('*') if f.is_file()}
    # Only prior failure diagnostics, not failed v1 data or any old train/weights.
    paths |= {f for f in r.previous.REPORT.glob('*') if f.is_file()}
    paths |= {r.REPORT / n for n in ('started.json', 'user-review-acceptance.json', 'handoff.md', 'verification.json', 'status.json', 'tests.log')}
    r.c.require(not (r.REPORT / 'failure.json').exists() and all(f.is_file() and f.resolve().is_relative_to(ROOT.resolve()) for f in paths), 'Incomplete/external archive sources')
    inventory = {f.relative_to(ROOT).as_posix(): {'bytes': f.stat().st_size, 'sha256': r.c.sha(f)} for f in sorted(paths)}
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for f in sorted(paths): archive.write(f, f.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        r.c.require(archive.testzip() is None and len(archive.namelist()) == len(set(archive.namelist())) and set(archive.namelist()) == set(inventory), 'CRC/member list differs')
        for name, item in inventory.items():
            r.c.require(hashlib.sha256(archive.read(name)).hexdigest() == item['sha256'], 'Archived bytes changed: ' + name)
        with tempfile.TemporaryDirectory(prefix='retention-release-1.1-') as tmp:
            extracted = Path(tmp) / 'data'; extracted.mkdir()
            for name in [*r.RELEASE_FILES, 'manifest.json', 'status.json']:
                with (extracted / name).open('xb') as f: f.write(archive.read((r.DATA / name).relative_to(ROOT).as_posix()))
            r.c.require(r.validate_release(extracted, gate) == verified, 'Extracted release data changed')
    delivered = {**verified, 'archive': archive_path.name, 'archive_bytes': archive_path.stat().st_size,
        'archive_sha256': r.c.sha(archive_path), 'tests_passed': 44, 'members': inventory,
        'kind': 'Incremental data release; prior unchanged candidate/framework-free dependencies required',
        'all_archive_member_hashes_checked': True, 'extracted_release_data_verification': 'Passed using unchanged dependency tree; not a clean-machine installation'}
    r.p.write(r.REPORT / 'delivery.json', delivered)
    r.p.write(r.REPORT / 'archive-verification.json', {k: v for k, v in delivered.items() if k != 'members'})
    print(json.dumps({k: v for k, v in delivered.items() if k != 'members'}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
