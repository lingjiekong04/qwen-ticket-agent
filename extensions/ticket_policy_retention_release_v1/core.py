"""Independent immutable data release. CPU simulation only; never train."""
from collections import Counter
from pathlib import Path
import shutil
import traceback
from extensions.ticket_policy_retention_build_v1 import core as c, pipeline as p
from scripts import record_ticket_policy_retention_ai_review_v1 as ai
from scripts import review_ticket_policy_retention_upload_v1 as upload

ROOT = c.ROOT
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'data/ticket_policy_retention_train_release_v1'
REPORT = ROOT / 'reports/ticket-policy-retention-release-v1'
REVIEW = upload.OUT
DATASET = 'ticket_policy_retention_train_release_v1'
QUOTAS = {'knowledge_chain': 480, 'authority': 360, 'clarification': 240,
          'faithful': 240, 'priority': 240, 'retention': 240}
POLICY = {'template': 'qwen', 'cutoff_len': 4096, 'train_on_prompt': False,
          'mask_history': False, 'packing': False}
KIND = 'user_acceptance_of_AI_assisted_sample_review'
PROVENANCE = 'Explicit current-chat user acceptance; not independently verified identity/signature or per-row human inspection'
EXPECTED = {
    'candidate': 'bd5290fbb7b68a8321ef238a0e96137333746b04ec803e64e8833e2c973d44bb',
    'targeted_eval': 'fca28aeab1eaf27f5d10bed8bcb0647d5d9e479bef09e26e679601d21fd5684b',
    'archive': 'd3375ae5f5197528d6bf8744286c6861af501f51c54c90a5f130bc926e194da7',
    'ai-semantic-review.json': 'e6c0c8feb41e7b0c49066adeff9a49ab3486807acd68faf0faf0a83eefed0ec8',
    'ai-review-notes.json': '00c28bac4040241701536392da13b652538554b60ca73cac84b02bc1e726f334',
    'evidence-review.json': 'a39a8e2d1d3b1c5e902f06b799d6007c3d455d438ba18985d9b2ef79cc8caad2',
}
COPY_FILES = {
    **{n: c.OUT / n for n in ('train.json', 'validation.json', 'coverage.json', 'isolation.json', 'supervision.json')},
    **{n: REVIEW / n for n in ('ai-semantic-review.json', 'ai-review-notes.json', 'evidence-review.json')},
    'token-audit.json': REVIEW / 'evidence/remote/report.json',
    'token-records.jsonl': REVIEW / 'evidence/remote/records.jsonl',
    'review-pairs.json': REVIEW / 'evidence/remote/review-pairs.json',
}
RELEASE_FILES = set(COPY_FILES) | {'dataset_info.json', 'user-review-acceptance.json'}


def sources():
    paths = [HERE / name for name in ('core.py', 'README.md', 'HANDOFF.md')]
    paths += [ROOT / name for name in (
        'scripts/release_ticket_policy_retention_v1.py',
        'scripts/package_ticket_policy_retention_release_v1.py',
        'tests/test_ticket_policy_retention_release_v1.py',
        'scripts/review_ticket_policy_retention_upload_v1.py',
        'scripts/record_ticket_policy_retention_ai_review_v1.py',
        'scripts/finalize_ticket_policy_retention_upload_review_v1.py',
        'tests/test_ticket_policy_retention_upload_review_v1.py')]
    return {path.relative_to(ROOT).as_posix(): c.sha(path) for path in paths}


def dataset_info():
    return {DATASET: c.dataset_info()['ticket_policy_retention_candidate_v1']}


def evidence_gate():
    for name in ('ai-semantic-review.json', 'ai-review-notes.json', 'evidence-review.json'):
        c.require(c.sha(REVIEW / name) == EXPECTED[name], 'Accepted review changed: ' + name)
    c.require(c.sha(c.OUT / 'manifest.json') == EXPECTED['candidate'] and c.sha(c.EVAL / 'manifest.json') == EXPECTED['targeted_eval'], 'Wrong accepted candidate or frozen evaluation')
    ev, sample, records = ai.gate(REVIEW)
    c.require(ev['archive_sha256'] == EXPECTED['archive'], 'Wrong uploaded CPU archive')
    # Replays/reconstructs all 1800 training + 192 frozen targeted gold episodes,
    # checks all records and sample selection; does not load tokenizer or weights.
    compared = upload.private_compare(REVIEW / 'evidence')
    c.require(compared == c.read(REVIEW / 'evidence/compare/report.json'), 'Evidence compare differs')
    review = c.read(REVIEW / 'ai-semantic-review.json')
    notes = c.read(REVIEW / 'ai-review-notes.json')
    status = c.read(REVIEW / 'status.json')
    check = c.read(REVIEW / 'verification.json')
    c.require(status['status'] == 'AI_review_complete_waiting_user_acceptance' and status['ai_review_sha256'] == EXPECTED['ai-semantic-review.json'] and status['notes_sha256'] == EXPECTED['ai-review-notes.json'] and status['report_sha256'] == c.sha(REVIEW / 'report.md'), 'Incomplete semantic review status')
    c.require(check['passed'] is True and check['tests_passed'] == 15 and check['test_log_sha256'] == c.sha(REVIEW / 'tests.log') and check['ai_review_sha256'] == EXPECTED['ai-semantic-review.json'], 'Missing review test evidence')
    for name, digest in check['sources_sha256'].items():
        c.require(name in sources() and c.sha(ROOT / name) == digest, 'Review source changed: ' + name)
    ids = [e['spec']['pair_id'] for e in sample[::2]]
    c.require(review['review_type'] == 'AI_assisted_semantic_sample_review' and review['reviewed_pairs'] == 60 and review['reviewed_rows'] == 120 and review['blocking_issues'] == [], 'Incomplete semantic conclusion')
    c.require([n['pair_id'] for n in notes] == ids and all(n['bilingual_request_and_answer_read'] is True and n['human_approved'] is False and n['decision'] in ('acceptable', 'acceptable_with_nonblocking_notes') for n in notes), 'Incomplete/overstated AI notes')
    c.require(review['notes_sha256'] == EXPECTED['ai-review-notes.json'] and review['evidence_review_sha256'] == EXPECTED['evidence-review.json'] and review['review_recorder_sha256'] == c.sha(Path(ai.__file__)), 'Review provenance mismatch')
    for note, zh, en in zip(notes, sample[::2], sample[1::2]):
        expected = [{'id': e['id'], 'episode_sha256': c.digest(e), 'row_sha256': c.digest(e['row']),
                     'events_sha256': c.digest(e['events']), 'final_state_sha256': c.digest(e['final_state']),
                     'input_ids_sha256': records[e['id']]['input_ids_sha256'], 'labels_sha256': records[e['id']]['labels_sha256']} for e in (zh, en)]
        c.require(note['row_evidence'] == expected, 'AI note input/label binding differs')
    for historical in (ev, review, status, check):
        c.require(historical['human_acceptance'] == 'pending' and historical['ready_for_training'] is False and historical['training_started'] is False, 'Historical stage scope changed')
    binding = {path.relative_to(ROOT).as_posix(): c.sha(path) for path in [
        c.OUT / 'manifest.json', c.EVAL / 'manifest.json',
        *[REVIEW / n for n in ('ai-semantic-review.json', 'ai-review-notes.json', 'evidence-review.json', 'verification.json', 'status.json', 'report.md', 'tests.log')],
        *[REVIEW / 'evidence' / n for n in sorted(upload.EXPECTED)]]}
    episodes = c.read(c.OUT / 'episodes.json')
    return c.read(c.OUT / 'manifest.json'), episodes, c.read(REVIEW / 'evidence/remote/report.json'), ids, binding


def acceptance(ids, binding):
    return {'date': '2026-10-08', 'user_message': '接受审核结论', 'decision': 'accepted',
            'kind': KIND, 'provenance': PROVENANCE,
            'scope': 'Accept the fixed 60 bilingual pairs AI-assisted sample review, including recorded nonblocking limitations; prepare independent data release only',
            'review_sample_pairs': 60, 'review_sample_rows': 120, 'pair_ids': ids,
            'bound_files_sha256': binding, 'AI_review_remains_AI_assisted': True,
            'per_row_human_inspection_attested': False, 'data_release_authorized': True,
            'training_authorized': False, 'gpu_inference_authorized': False, 'training_started': False}


def validate_release(directory, gate):
    cm, episodes, remote, ids, binding = gate
    c.require(not (directory / 'failure.json').exists(), 'Failed release; preserve diagnosis, do not use')
    manifest = c.read(directory / 'manifest.json')
    status = c.read(directory / 'status.json')
    c.require(status['status'] == 'complete' and status['manifest_sha256'] == c.sha(directory / 'manifest.json'), 'Incomplete release manifest')
    c.require(manifest['kind'] == 'independent_training_data_release' and manifest['dataset'] == DATASET, 'Wrong dataset release')
    for key in ('ready_for_training', 'real_tokenizer_audited'):
        c.require(manifest[key] is True, 'Release gate incomplete: ' + key)
    for key in ('training_authorized', 'training_started', 'gpu_inference_authorized', 'gpu_inference_started', 'model_weights_loaded', 'old_final_test_read', 'per_row_human_inspection_attested'):
        c.require(manifest[key] is False, 'Scope overstated: ' + key)
    c.require(manifest['rows'] == 1800 and manifest['pairs'] == 900 and manifest['locales'] == {'zh': 900, 'en': 900} and manifest['quotas_rows'] == QUOTAS, 'Release quotas changed')
    c.require(manifest['review_sample_pairs'] == 60 and manifest['review_sample_rows'] == 120 and manifest['review_kind'] == KIND and manifest['review_provenance'] == PROVENANCE, 'Review scope changed')
    c.require(manifest['candidate_manifest_sha256'] == EXPECTED['candidate'] and manifest['targeted_eval_manifest_sha256'] == EXPECTED['targeted_eval'] and manifest['seed'] == cm['seed'] and manifest['candidate_sources_sha256'] == cm['sources_sha256'] and manifest['candidate_dependencies_sha256'] == cm['dependencies_sha256'] and manifest['review_binding_sha256'] == binding and manifest['release_sources_sha256'] == sources(), 'Release provenance changed')
    c.require(manifest['supervision'] == POLICY and all(type(manifest['supervision'][k]) is bool for k in ('train_on_prompt', 'mask_history', 'packing')), 'Supervision changed')
    c.require(set(manifest['files_sha256']) == RELEASE_FILES, 'Unsafe or incomplete release file bindings')
    for name, digest in manifest['files_sha256'].items():
        c.require(c.sha(directory / name) == digest, 'Release file changed: ' + name)
    for name, source in COPY_FILES.items():
        c.require(c.sha(directory / name) == c.sha(source), 'Not byte-identical candidate/evidence copy: ' + name)
    human = c.read(directory / 'user-review-acceptance.json')
    c.require(human == acceptance(ids, binding) and human == c.read(REPORT / 'user-review-acceptance.json'), 'Acceptance missing or overstated')
    c.require(c.read(directory / 'dataset_info.json') == dataset_info(), 'ShareGPT roles or registration changed')
    rows = c.read(directory / 'train.json')
    c.require(rows == [e['row'] for e in episodes] and len(rows) == len({c.digest(r) for r in rows}) == 1800 and all(set(r) == {'system', 'tools', 'conversations'} for r in rows), 'Projection, count or hidden metadata changed')
    c.require(Counter(e['spec']['theme'] for e in episodes) == QUOTAS and Counter(e['locale'] for e in episodes) == {'zh': 900, 'en': 900}, 'Candidate population differs')
    validation = c.read(directory / 'validation.json')
    c.require(validation['passed'] is True and len(validation['records']) == 1800 and all(r['passed'] is True for r in validation['records']) and validation['independent_replay_rows'] == validation['deterministic_reconstruction_rows'] == 1800, 'Full replay evidence missing')
    c.require(c.read(directory / 'isolation.json')['passed'] is True and c.read(directory / 'supervision.json') == POLICY, 'Isolation or supervision incomplete')
    return {'passed': True, 'rows': 1800, 'pairs': 900, 'quotas_rows': QUOTAS,
            'locales': {'zh': 900, 'en': 900}, 'max_tokens': remote['max_tokens'],
            'assistant_turns': remote['assistant_turns'], 'review_sample_pairs': 60, 'review_sample_rows': 120,
            'review_kind': KIND, 'review_provenance': PROVENANCE, 'per_row_human_inspection_attested': False,
            'token_audit': 'Verified uploaded remote CPU audit; no independent local tokenizer rerun',
            'release_manifest_sha256': c.sha(directory / 'manifest.json'),
            'candidate_manifest_sha256': EXPECTED['candidate'], 'targeted_eval_manifest_sha256': EXPECTED['targeted_eval'],
            'ready_for_training': True, 'training_authorized': False, 'training_started': False, 'gpu_inference_started': False,
            'old_final_test_read': False}


def verify(directory=DATA):
    return validate_release(directory, evidence_gate())


def build(user_message):
    c.require(user_message == '接受审核结论', 'Explicit current-chat review acceptance required')
    p.fresh(DATA); p.fresh(REPORT)
    REPORT.mkdir(parents=True)
    p.write(REPORT / 'started.json', {'stage': 'data_release', 'started_at': p.now(), 'training_started': False})
    try:
        gate = evidence_gate()
        cm, _, _, ids, binding = gate
        receipt = acceptance(ids, binding)
        p.write(REPORT / 'user-review-acceptance.json', receipt)
        shutil.copyfile(HERE / 'HANDOFF.md', REPORT / 'handoff.md')
        DATA.mkdir(parents=True)
        p.write(DATA / 'started.json', {'stage': 'data_release', 'started_at': p.now(), 'training_started': False})
        for name, source in COPY_FILES.items():
            # All destinations belong to the newly created exclusive directory.
            with source.open('rb') as src, (DATA / name).open('xb') as dst:
                shutil.copyfileobj(src, dst)
        p.write(DATA / 'dataset_info.json', dataset_info())
        p.write(DATA / 'user-review-acceptance.json', receipt)
        p.write(DATA / 'manifest.json', {
            'kind': 'independent_training_data_release', 'dataset': DATASET, 'released_at': p.now(),
            'seed': cm['seed'], 'rows': 1800, 'pairs': 900, 'locales': {'zh': 900, 'en': 900}, 'quotas_rows': QUOTAS,
            'candidate_manifest_sha256': EXPECTED['candidate'], 'targeted_eval_manifest_sha256': EXPECTED['targeted_eval'],
            'candidate_sources_sha256': cm['sources_sha256'], 'candidate_dependencies_sha256': cm['dependencies_sha256'],
            'review_binding_sha256': binding, 'release_sources_sha256': sources(),
            'files_sha256': {name: c.sha(DATA / name) for name in sorted(RELEASE_FILES)},
            'supervision': POLICY, 'review_kind': KIND, 'review_provenance': PROVENANCE,
            'review_sample_pairs': 60, 'review_sample_rows': 120, 'per_row_human_inspection_attested': False,
            'real_tokenizer_audited': True, 'ready_for_training': True,
            'training_authorized': False, 'training_started': False, 'gpu_inference_authorized': False,
            'gpu_inference_started': False, 'model_weights_loaded': False, 'old_final_test_read': False})
        p.write(DATA / 'status.json', {'status': 'complete', 'manifest_sha256': c.sha(DATA / 'manifest.json'), 'training_started': False})
        result = validate_release(DATA, gate)
        p.write(REPORT / 'verification.json', result)
        p.write(REPORT / 'status.json', {'status': 'complete', 'verification_sha256': c.sha(REPORT / 'verification.json'), 'training_started': False})
        return result
    except BaseException as exc:
        p.text(REPORT / 'failure.log', traceback.format_exc())
        failure = {'status': 'failed', 'error': repr(exc), 'ready_for_training': False, 'training_started': False}
        p.write(REPORT / 'failure.json', failure)
        if DATA.exists():
            p.write(DATA / 'failure.json', failure)
        raise
