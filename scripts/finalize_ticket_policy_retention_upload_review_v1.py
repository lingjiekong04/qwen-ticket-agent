"""Append test evidence and package review only; never release training data."""
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import record_ticket_policy_retention_ai_review_v1 as a
from extensions.ticket_policy_retention_build_v1 import core as c, pipeline as p


def main():
    out = a.OUT
    archive = ROOT / 'lora_sft_ticket_policy_retention_upload_review_v1.zip'
    for file in (out / 'tests.log', out / 'verification.json', archive, out / 'delivery.json'):
        p.fresh(file)
    evidence, sample, _ = a.gate(out)
    review = c.read(out / 'ai-semantic-review.json')
    notes = c.read(out / 'ai-review-notes.json')
    status = c.read(out / 'status.json')
    c.require(review['notes_sha256'] == c.sha(out / 'ai-review-notes.json') and review['review_recorder_sha256'] == c.sha(Path(a.__file__)), 'AI note/source changed')
    c.require(review['evidence_review_sha256'] == c.sha(out / 'evidence-review.json') and status['ai_review_sha256'] == c.sha(out / 'ai-semantic-review.json') and status['report_sha256'] == c.sha(out / 'report.md'), 'AI evidence/report changed')
    c.require([n['pair_id'] for n in notes] == evidence['sample_pair_ids'] and len(notes) == 60 and len(sample) == 120, 'Incomplete semantic notes')
    c.require(review['human_acceptance'] == 'pending' and review['training_authorized'] is False and not review['blocking_issues'], 'Unexpected approval or findings')
    manifest = c.read(c.OUT / 'manifest.json')
    c.require(manifest['sources_sha256'] == p.sources() and manifest['dependencies_sha256'] == p.dependencies(), 'Old bound sources changed')
    for name, digest in manifest['files_sha256'].items():
        c.require(c.sha(c.OUT / name) == digest, 'Candidate file changed: ' + name)
    eval_manifest = c.read(c.EVAL / 'manifest.json')
    for name, digest in eval_manifest['files_sha256'].items():
        c.require(c.sha(c.EVAL / name) == digest, 'Frozen targeted evaluation changed: ' + name)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', TEMP=str(ROOT / '.t'), TMP=str(ROOT / '.t'))
    test = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'tests.test_ticket_policy_retention_upload_review_v1', '-v'], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.text(out / 'tests.log', test.stdout.decode('utf-8', errors='replace'))
    c.require(test.returncode == 0, 'Review gate tests failed; log preserved')
    sources = [Path(a.__file__), ROOT / 'scripts/review_ticket_policy_retention_upload_v1.py', Path(__file__), ROOT / 'tests/test_ticket_policy_retention_upload_review_v1.py']
    p.write(out / 'verification.json', {
        'passed': True, 'tests_passed': 15, 'test_log_sha256': c.sha(out / 'tests.log'),
        'AI_reviewed_pairs': 60, 'AI_reviewed_rows': 120, 'blocking_issues': [],
        'candidate_and_frozen_eval_files_unchanged': True, 'old_bound_sources_unchanged': True,
        'sources_sha256': {f.relative_to(ROOT).as_posix(): c.sha(f) for f in sources},
        'ai_review_sha256': c.sha(out / 'ai-semantic-review.json'),
        'human_acceptance': 'pending', 'per_row_human_inspection_attested': False,
        'ready_for_training': False, 'training_started': False, 'gpu_inference_started': False,
        'old_final_test_read': False,
    })
    members = sorted([*sources, *[f for f in out.rglob('*') if f.is_file()]])
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for f in members:
            z.write(f, f.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        c.require(z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(members), 'Review archive verification failed')
    p.write(out / 'delivery.json', {
        'archive': archive.name, 'archive_sha256': c.sha(archive),
        'members_sha256': {f.relative_to(ROOT).as_posix(): c.sha(f) for f in members},
        'scope': 'Review evidence only; not a training release or authorization',
        'human_acceptance': 'pending', 'training_started': False,
    })
    print({'passed': True, 'reviewed_pairs': 60, 'tests': 15, 'human_acceptance': 'pending', 'training_started': False, 'archive': archive.name})


if __name__ == '__main__':
    main()
