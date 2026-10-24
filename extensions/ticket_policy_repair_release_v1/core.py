"""Append-only data release gates. CPU replay only; no model or training entrypoint."""
from collections import Counter
from pathlib import Path
import shutil
from extensions.ticket_policy_repair_build_v1 import core as c
from extensions.ticket_policy_repair_build_v1.pipeline import verify as candidate_verify, dataset_info as candidate_info
from scripts import review_ticket_policy_repair_upload_v1 as upload

s = c.s
ROOT = c.ROOT
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'data/ticket_policy_repair_train_release_v1'
REPORT = ROOT / 'reports/ticket-policy-repair-release-v1'
REVIEW = c.REPORT / 'upload-review-20261007'
DATASET = 'ticket_policy_repair_train_release_v1'
POLICY = {'template': 'qwen', 'cutoff_len': 4096, 'train_on_prompt': False,
          'mask_history': False, 'packing': False}
ACCEPTANCE_KIND = 'user_acceptance_of_ai_assisted_sample_review'
REVIEW_PROVENANCE = 'Explicit current-chat user acceptance; not independently verified identity/signature or per-row human inspection'
EXPECTED = {
    'candidate_manifest': 'bd52385b66a4a71bdc4df904ea74a5c002a1edcfcea49de50d2ed85c0a4227c3',
    'archive': '58a8244152dbaa0bb9763a941077f34730ee6b2073130c8a04620926d05ab2ea',
    'ai-semantic-review.json': 'e7141a7ad08003be50a93b20c96e5afc7075be9ccbbe0b7ec2134b4a31c486bb',
    'ai-review.md': '0797935b748b8135aef9187a52b80e301a3cdd29236c08fc383d9b28936c2e42',
    'evidence-review.json': 'e3abce1d8c9c6ab7ec6bff61cdcc7098389668e0614cfc518ab68422efbcf871',
}
COPY_FILES = {
    'train.json': c.OUT / 'train.json',
    'validation.json': c.OUT / 'validation.json',
    'coverage.json': c.OUT / 'coverage.json',
    'isolation-similarity.json': c.OUT / 'isolation-similarity.json',
    'token-audit.json': REVIEW / 'evidence/remote/report.json',
    'token-records.jsonl': REVIEW / 'evidence/remote/records.jsonl',
    'review-pairs.json': REVIEW / 'evidence/remote/review-pairs.json',
    'ai-semantic-review.json': REVIEW / 'ai-semantic-review.json',
    'evidence-review.json': REVIEW / 'evidence-review.json',
}
RELEASE_FILES = set(COPY_FILES) | {'dataset_info.json', 'user-review-acceptance.json'}


def dataset_info():
    return {DATASET: candidate_info()['ticket_policy_repair_candidate_v1']}


def sources():
    paths = [HERE / 'core.py', HERE / 'README.md', HERE / 'HANDOFF.md', ROOT / 'scripts/release_ticket_policy_repair_v1.py',
             ROOT / 'scripts/package_ticket_policy_repair_release_v1.py',
             ROOT / 'scripts/review_ticket_policy_repair_upload_v1.py',
             ROOT / 'tests/test_policy_repair_release_v1.py', ROOT / 'tests/test_policy_repair_uploaded_evidence_v1.py']
    return {p.relative_to(ROOT).as_posix(): s.sha(p) for p in paths}


def evidence_gate():
    cm, episodes = candidate_verify()  # Independently replay the full population, never skipped by CLI.
    s.require(s.sha(c.OUT / 'manifest.json') == EXPECTED['candidate_manifest'], 'Wrong accepted candidate')
    for name in ('ai-semantic-review.json', 'ai-review.md', 'evidence-review.json'):
        s.require(s.sha(REVIEW / name) == EXPECTED[name], 'Accepted review evidence changed: ' + name)
    review_status = s.read(REVIEW / 'status.json')
    ev = s.read(REVIEW / 'evidence-review.json')
    ai = s.read(REVIEW / 'ai-semantic-review.json')
    s.require(review_status['status'] == 'complete' and review_status['evidence_review_sha256'] == EXPECTED['evidence-review.json'], 'Incomplete uploaded review')
    files = {name: (REVIEW / 'evidence' / name).read_bytes() for name in upload.EXPECTED}
    s.require(ev['member_sha256'] == {k: upload.sha(v) for k, v in files.items()}
              and ev['archive_sha256'] == ai['archive_sha256'] == EXPECTED['archive']
              and ev['candidate_manifest_sha256'] == ai['candidate_manifest_sha256'] == EXPECTED['candidate_manifest'], 'Uploaded review source differs')
    remote, records, sample = upload.check_evidence(files, episodes)
    ids = [e['spec']['pair_id'] for e in sample if e['locale'] == 'zh']
    s.require(ev['passed'] is True and ev['rows'] == 1200 and ev['sample_pair_ids'] == ids
              and ev['local_candidate_replay'] == 'passed_all_1200' and ev['formal_sample_recomputed'] == 'matched', 'Incomplete evidence checks')
    s.require(ai['conclusion'] == 'no_blocking_issue_found_in_sample' and ai['sample_pairs_read'] == 48
              and ai['sample_episodes_read'] == 96 and [p['pair_id'] for p in ai['pairs']] == ids
              and all(p['both_locales'] == 'no_blocking_issue' for p in ai['pairs']), 'AI sample decision incomplete')
    s.require(ai['sample_episodes_sha256'] == upload.sha(files['remote/review-episodes.json'])
              and ai['remote_report_sha256'] == upload.sha(files['remote/report.json'])
              and ai['records_sha256'] == upload.sha(files['remote/records.jsonl']), 'AI decision bound to different evidence')
    for previous in (ev, ai):
        s.require(previous['human_review'] == 'pending' and previous['ready_for_training'] is False
                  and previous['training_started'] is False, 'Historical audit scope changed')
    binding = {p.relative_to(ROOT).as_posix(): s.sha(p) for p in [
        c.OUT / 'manifest.json', REVIEW / 'ai-semantic-review.json', REVIEW / 'ai-review.md',
        REVIEW / 'evidence-review.json', REVIEW / 'evidence/remote/report.json',
        REVIEW / 'evidence/remote/records.jsonl', REVIEW / 'evidence/remote/review-pairs.json',
        REVIEW / 'evidence/remote/review-episodes.json', REVIEW / 'evidence/compare/report.json']}
    return cm, episodes, remote, ids, binding


def acceptance_record(ids, binding):
    return {'date': '2026-10-07', 'user_message': '接受', 'decision': 'accepted',
            'kind': ACCEPTANCE_KIND, 'provenance': REVIEW_PROVENANCE,
            'scope': 'Accept the AI-assisted review of these 48 bilingual sample pairs and prepare the independent data release only',
            'sample_pairs': 48, 'sample_rows': 96, 'pair_ids': ids, 'bound_files_sha256': binding,
            'ai_review_remains_ai_assisted': True, 'per_row_human_inspection_attested': False,
            'data_release_authorized': True, 'training_authorized': False, 'training_started': False}


def verify(directory=DATA):
    cm, episodes, remote, ids, binding = evidence_gate()
    manifest = s.read(directory / 'manifest.json')
    status = s.read(directory / 'status.json')
    s.require(status['status'] == 'complete' and status['manifest_sha256'] == s.sha(directory / 'manifest.json'), 'Incomplete release')
    s.require(manifest['kind'] == 'independent_training_data_release' and manifest['dataset'] == DATASET
              and manifest['ready_for_training'] is True and manifest['real_tokenizer_audited'] is True, 'Not a verified data release')
    s.require(manifest['training_authorized'] is False and manifest['training_started'] is False
              and manifest['model_weights_loaded'] is False and manifest['old_final_test_read'] is False, 'Release scope changed')
    s.require(manifest['candidate_manifest_sha256'] == EXPECTED['candidate_manifest']
              and manifest['revision'] == cm['revision'] and manifest['seed'] == cm['seed']
              and manifest['candidate_sources'] == cm['sources'] and manifest['binding'] == cm['binding']
              and manifest['isolation'] == cm['isolation'] and manifest['release_sources'] == sources(), 'Release provenance changed')
    s.require(manifest['training_policy'] == POLICY and all(type(manifest['training_policy'][k]) is bool for k in ('train_on_prompt', 'mask_history', 'packing')), 'Supervision policy changed')
    s.require(manifest['rows'] == 1200 and manifest['pairs'] == 600 and manifest['locales'] == {'zh': 600, 'en': 600}
              and manifest['quotas'] == c.QUOTAS and manifest['review_sample_pairs'] == 48
              and manifest['review_sample_rows'] == 96 and manifest['human_review'] == ACCEPTANCE_KIND, 'Count/review mismatch')
    s.require(set(manifest['files_sha256']) == RELEASE_FILES, 'Release file list incomplete or unsafe')
    for name, expected in manifest['files_sha256'].items():
        s.require(s.sha(directory / name) == expected, 'Released file changed: ' + name)
    for name, origin in COPY_FILES.items():
        s.require(s.sha(directory / name) == s.sha(origin), 'Not a byte-identical release copy: ' + name)
    human = s.read(directory / 'user-review-acceptance.json')
    s.require(human == acceptance_record(ids, binding) and human == s.read(REPORT / 'user-review-acceptance.json'), 'Acceptance missing, changed, or overstated')
    s.require(s.read(directory / 'dataset_info.json') == dataset_info(), 'ShareGPT registration changed')
    rows = s.read(directory / 'train.json')
    s.require(rows == [e['row'] for e in episodes] and len({s.digest(r) for r in rows}) == 1200
              and all(set(r) == {'system', 'tools', 'conversations'} for r in rows), 'Training projection/hidden metadata changed')
    s.require(Counter(e['spec']['category'] for e in episodes) == c.QUOTAS
              and Counter(e['locale'] for e in episodes) == {'zh': 600, 'en': 600}, 'Population quotas changed')
    validation = s.read(directory / 'validation.json')
    s.require(validation['passed'] is True and len(validation['records']) == 1200
              and all(r['passed'] is True for r in validation['records']), 'Full replay missing')
    isolation = s.read(directory / 'isolation-similarity.json')
    s.require(isolation['flags'] == [], 'Unresolved isolation screening flags')
    return {'passed': True, 'rows': 1200, 'pairs': 600, 'quotas': c.QUOTAS, 'locales': {'zh': 600, 'en': 600},
            'max_tokens': remote['max_tokens'], 'assistant_turns': remote['assistant_turns'],
            'review_sample_pairs': 48, 'review_sample_rows': 96, 'human_review': ACCEPTANCE_KIND,
            'review_provenance': REVIEW_PROVENANCE, 'per_row_human_inspection_attested': False,
            'token_audit': 'Verified uploaded remote CPU audit; no independent local tokenizer rerun',
            'release_manifest_sha256': s.sha(directory / 'manifest.json'),
            'candidate_manifest_sha256': EXPECTED['candidate_manifest'],
            'ready_for_training': True, 'training_authorized': False, 'training_started': False}


def build(user_acceptance):
    s.require(user_acceptance == '接受', 'Explicit current-chat acceptance required')
    s.fresh(DATA)
    s.fresh(REPORT)
    REPORT.mkdir(parents=True)
    s.write(REPORT / 'status.json', {'status': 'started', 'training_started': False})
    try:
        cm, episodes, remote, ids, binding = evidence_gate()
        accepted = acceptance_record(ids, binding)
        s.write(REPORT / 'user-review-acceptance.json', accepted)
        shutil.copyfile(HERE / 'HANDOFF.md', REPORT / 'handoff.md')
        DATA.mkdir(parents=True)
        s.write(DATA / 'status.json', {'status': 'started', 'training_started': False})
        for name, origin in COPY_FILES.items():
            shutil.copyfile(origin, DATA / name)
        s.write(DATA / 'dataset_info.json', dataset_info())
        shutil.copyfile(REPORT / 'user-review-acceptance.json', DATA / 'user-review-acceptance.json')
        s.write(DATA / 'manifest.json', {'kind': 'independent_training_data_release', 'dataset': DATASET,
            'revision': cm['revision'], 'seed': cm['seed'], 'rows': 1200, 'pairs': 600,
            'locales': {'zh': 600, 'en': 600}, 'quotas': c.QUOTAS,
            'candidate_manifest_sha256': EXPECTED['candidate_manifest'], 'candidate_sources': cm['sources'],
            'binding': cm['binding'], 'isolation': cm['isolation'], 'release_sources': sources(),
            'files_sha256': {name: s.sha(DATA / name) for name in sorted(RELEASE_FILES)},
            'training_policy': POLICY, 'human_review': ACCEPTANCE_KIND, 'review_provenance': REVIEW_PROVENANCE,
            'review_sample_pairs': 48, 'review_sample_rows': 96, 'per_row_human_inspection_attested': False,
            'real_tokenizer_audited': True, 'ready_for_training': True,
            'training_authorized': False, 'training_started': False, 'model_weights_loaded': False, 'old_final_test_read': False})
        s.update(DATA / 'status.json', {'status': 'complete', 'manifest_sha256': s.sha(DATA / 'manifest.json'), 'training_started': False})
        result = verify()
        s.write(REPORT / 'verification.json', result)
        s.update(REPORT / 'status.json', {'status': 'complete', 'verification_sha256': s.sha(REPORT / 'verification.json'), 'training_started': False})
        return result
    except BaseException as exc:
        failure = {'status': 'failed', 'error': repr(exc), 'ready_for_training': False, 'training_started': False}
        s.update(REPORT / 'status.json', failure)
        if DATA.exists(): s.update(DATA / 'status.json', failure)
        raise
