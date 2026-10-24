"""Release unchanged 1800 episodes with explicitly accepted duplicate disclosure."""
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import traceback
from extensions.ticket_policy_retention_release_v1 import core as previous

c, p, ROOT, REVIEW = previous.c, previous.p, previous.ROOT, previous.REVIEW
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'data/ticket_policy_retention_train_release_v1_1'
REPORT = ROOT / 'reports/ticket-policy-retention-release-v1.1'
DATASET = 'ticket_policy_retention_train_release_v1_1'
QUOTAS, POLICY, KIND, PROVENANCE = previous.QUOTAS, previous.POLICY, previous.KIND, previous.PROVENANCE
EXPECTED = dict(previous.EXPECTED)
DISCLOSURE = previous.REPORT / 'public-row-duplicates.json'
DISCLOSURE_SHA = '20aff058d4b3ccc75b9c909bbd4361a00c241b15060610749e76f31631f93c67'
PREVIOUS_GATE_SHA = '4018af439014e1ee5b2035951ac973313294996893f149e8deab46e8c5427e43'
COPY_FILES = {**previous.COPY_FILES, 'public-row-duplicates.json': DISCLOSURE}
RELEASE_FILES = set(COPY_FILES) | {'dataset_info.json', 'user-review-acceptance.json'}


def sources():
    paths = [HERE / n for n in ('core.py', 'README.md', 'HANDOFF.md')]
    paths += [ROOT / n for n in ('scripts/release_ticket_policy_retention_v1_1.py',
        'scripts/package_ticket_policy_retention_release_v1_1.py', 'tests/test_ticket_policy_retention_release_v1_1.py')]
    return {**previous.sources(), **{path.relative_to(ROOT).as_posix(): c.sha(path) for path in paths}}


def dataset_info():
    return {DATASET: c.dataset_info()['ticket_policy_retention_candidate_v1']}


def duplicate_metrics(episodes):
    grouped = defaultdict(list)
    for e in episodes:
        grouped[c.digest(e['row'])].append(e)
    groups = []
    for digest, es in sorted(grouped.items()):
        if len(es) <= 1:
            continue
        groups.append({'public_row_sha256': digest, 'episode_ids': [e['id'] for e in es],
            'themes': [e['spec']['theme'] for e in es], 'variants': [e['spec']['variant'] for e in es],
            'locales': [e['locale'] for e in es], 'contrast_groups': [e['spec']['contrast_group'] for e in es],
            'permission_profiles': [e['spec']['permission_profile'] for e in es],
            'zero_tool_calls': all(not e['events'] for e in es),
            'all_permission_denied': all(e['spec']['allow_write'] is False for e in es),
            'distinct_full_episodes': len({c.digest(e) for e in es}) == len(es),
            'distinct_fixture_hashes': [c.digest(e['spec']['fixture']) for e in es]})
    return {'unique_public_train_rows': len(grouped), 'duplicate_public_row_groups': len(groups),
            'excess_public_rows': len(episodes) - len(grouped),
            'max_public_row_multiplicity': max(map(len, grouped.values())), 'groups': groups}


def check_duplicates(episodes):
    c.require(c.sha(DISCLOSURE) == DISCLOSURE_SHA, 'Accepted duplicate disclosure changed')
    disclosed = c.read(DISCLOSURE)
    actual = duplicate_metrics(episodes)
    for key, value in actual.items():
        c.require(disclosed[key] == value, 'Duplicate group/membership differs: ' + key)
    c.require((actual['unique_public_train_rows'], actual['duplicate_public_row_groups'], actual['excess_public_rows'], actual['max_public_row_multiplicity']) == (1762, 38, 38, 2), 'Unexpected new duplicate population')
    c.require(disclosed['candidate_manifest_sha256'] == EXPECTED['candidate'] and disclosed['public_train_sha256'] == c.sha(c.OUT / 'train.json') and disclosed['full_episodes_sha256'] == c.sha(c.OUT / 'episodes.json'), 'Duplicate source differs')
    c.require(len({e['id'] for e in episodes}) == len({c.digest(e) for e in episodes}) == 1800, 'Full episodes changed or duplicated')
    return {k: v for k, v in actual.items() if k != 'groups'}


def evidence_gate():
    c.require(c.sha(Path(previous.__file__)) == PREVIOUS_GATE_SHA, 'Previous CPU evidence gate changed')
    gate = previous.evidence_gate()  # Full immutable CPU replay, no tokenizer/model.
    check_duplicates(gate[1])
    return gate


def acceptance(ids, binding):
    return {'date': '2026-10-08', 'user_message': '保留原数据发布', 'prior_user_message': '接受审核结论',
            'decision': 'accepted', 'kind': KIND, 'provenance': PROVENANCE,
            'scope': 'Accept the fixed 60-pair AI-assisted review and the explicit 1762 unique public rows / 38 excess rows disclosure; retain all original 1800 episodes and quotas, release only',
            'review_sample_pairs': 60, 'review_sample_rows': 120, 'pair_ids': ids,
            'bound_files_sha256': binding, 'duplicate_disclosure_sha256': DISCLOSURE_SHA,
            'duplicate_handling': 'keep_original_rows_without_deduplication',
            'AI_review_remains_AI_assisted': True, 'per_row_human_inspection_attested': False,
            'data_release_authorized': True, 'training_authorized': False,
            'gpu_inference_authorized': False, 'training_started': False}


def validate_release(directory, gate, *, require_completion=True):
    cm, episodes, remote, ids, binding = gate
    c.require(not (directory / 'failure.json').exists(), 'Failed release; preserve diagnosis, do not use')
    manifest = c.read(directory / 'manifest.json')
    if require_completion:
        status = c.read(directory / 'status.json')
        c.require(status['status'] == 'complete' and status['manifest_sha256'] == c.sha(directory / 'manifest.json'), 'Incomplete release manifest')
    c.require(manifest['kind'] == 'independent_training_data_release' and manifest['revision'] == '1.1' and manifest['dataset'] == DATASET, 'Wrong release revision/dataset')
    for key in ('ready_for_training', 'real_tokenizer_audited'):
        c.require(manifest[key] is True, 'Incomplete release gate: ' + key)
    for key in ('training_authorized', 'training_started', 'gpu_inference_authorized', 'gpu_inference_started', 'model_weights_loaded', 'old_final_test_read', 'per_row_human_inspection_attested'):
        c.require(manifest[key] is False, 'Scope overstated: ' + key)
    c.require(manifest['rows'] == 1800 and manifest['pairs'] == 900 and manifest['locales'] == {'zh': 900, 'en': 900} and manifest['quotas_rows'] == QUOTAS, 'Counts/quotas changed')
    c.require(manifest['review_sample_pairs'] == 60 and manifest['review_sample_rows'] == 120 and manifest['review_kind'] == KIND and manifest['review_provenance'] == PROVENANCE, 'Review scope changed')
    c.require(manifest['candidate_manifest_sha256'] == EXPECTED['candidate'] and manifest['targeted_eval_manifest_sha256'] == EXPECTED['targeted_eval'] and manifest['seed'] == cm['seed'] and manifest['candidate_sources_sha256'] == cm['sources_sha256'] and manifest['candidate_dependencies_sha256'] == cm['dependencies_sha256'] and manifest['review_binding_sha256'] == binding and manifest['release_sources_sha256'] == sources(), 'Provenance changed')
    c.require(manifest['supervision'] == POLICY and all(type(manifest['supervision'][k]) is bool for k in ('train_on_prompt', 'mask_history', 'packing')), 'Supervision changed')
    c.require(set(manifest['files_sha256']) == RELEASE_FILES, 'Unsafe/incomplete release file bindings')
    for name, digest in manifest['files_sha256'].items():
        c.require(c.sha(directory / name) == digest, 'Release file changed: ' + name)
    for name, source in COPY_FILES.items():
        c.require(c.sha(directory / name) == c.sha(source), 'Not byte-identical source copy: ' + name)
    human = c.read(directory / 'user-review-acceptance.json')
    c.require(human == acceptance(ids, binding) and human == c.read(REPORT / 'user-review-acceptance.json'), 'Acceptance missing/overstated')
    c.require(c.read(directory / 'dataset_info.json') == dataset_info(), 'ShareGPT registration/roles changed')
    rows = c.read(directory / 'train.json')
    c.require(rows == [e['row'] for e in episodes] and len(rows) == 1800 and all(set(r) == {'system', 'tools', 'conversations'} for r in rows), 'Projection, counts or hidden metadata changed')
    duplicate = check_duplicates(episodes)
    c.require(manifest['duplicate_metrics'] == duplicate and manifest['duplicate_disclosure_sha256'] == DISCLOSURE_SHA and manifest['duplicate_handling'] == 'keep_original_rows_without_deduplication', 'Accepted duplicate policy/counts changed')
    c.require(Counter(e['spec']['theme'] for e in episodes) == QUOTAS and Counter(e['locale'] for e in episodes) == {'zh': 900, 'en': 900}, 'Population quotas differ')
    validation = c.read(directory / 'validation.json')
    c.require(validation['passed'] is True and len(validation['records']) == 1800 and all(x['passed'] is True for x in validation['records']) and validation['independent_replay_rows'] == validation['deterministic_reconstruction_rows'] == 1800, 'Missing full replay')
    c.require(c.read(directory / 'isolation.json')['passed'] is True and c.read(directory / 'supervision.json') == POLICY, 'Isolation or labels policy incomplete')
    return {'passed': True, 'revision': '1.1', 'rows': 1800, 'pairs': 900, 'quotas_rows': QUOTAS,
            'locales': {'zh': 900, 'en': 900}, 'max_tokens': remote['max_tokens'], 'assistant_turns': remote['assistant_turns'],
            'review_sample_pairs': 60, 'review_sample_rows': 120, 'review_kind': KIND, 'review_provenance': PROVENANCE,
            'per_row_human_inspection_attested': False, 'duplicate_metrics': duplicate,
            'duplicate_handling': 'keep_original_rows_without_deduplication',
            'token_audit': 'Verified uploaded remote CPU audit; no independent local tokenizer rerun',
            'release_manifest_sha256': c.sha(directory / 'manifest.json'), 'candidate_manifest_sha256': EXPECTED['candidate'],
            'targeted_eval_manifest_sha256': EXPECTED['targeted_eval'], 'ready_for_training': True,
            'training_authorized': False, 'training_started': False, 'gpu_inference_started': False, 'old_final_test_read': False}


def verify(directory=DATA):
    return validate_release(directory, evidence_gate())


def build(user_message):
    c.require(user_message == '保留原数据发布', 'Explicit duplicate retention acceptance required')
    p.fresh(DATA); p.fresh(REPORT)
    REPORT.mkdir(parents=True)
    p.write(REPORT / 'started.json', {'stage': 'data_release_1.1', 'started_at': p.now(), 'training_started': False})
    try:
        gate = evidence_gate()
        cm, episodes, _, ids, binding = gate
        receipt = acceptance(ids, binding)
        p.write(REPORT / 'user-review-acceptance.json', receipt)
        shutil.copyfile(HERE / 'HANDOFF.md', REPORT / 'handoff.md')
        DATA.mkdir(parents=True)
        p.write(DATA / 'started.json', {'stage': 'data_release_1.1', 'started_at': p.now(), 'training_started': False})
        for name, source in COPY_FILES.items():
            with source.open('rb') as src, (DATA / name).open('xb') as dst:
                shutil.copyfileobj(src, dst)
        p.write(DATA / 'dataset_info.json', dataset_info())
        p.write(DATA / 'user-review-acceptance.json', receipt)
        p.write(DATA / 'manifest.json', {
            'kind': 'independent_training_data_release', 'revision': '1.1', 'dataset': DATASET, 'released_at': p.now(),
            'seed': cm['seed'], 'rows': 1800, 'pairs': 900, 'locales': {'zh': 900, 'en': 900}, 'quotas_rows': QUOTAS,
            'candidate_manifest_sha256': EXPECTED['candidate'], 'targeted_eval_manifest_sha256': EXPECTED['targeted_eval'],
            'candidate_sources_sha256': cm['sources_sha256'], 'candidate_dependencies_sha256': cm['dependencies_sha256'],
            'review_binding_sha256': binding, 'release_sources_sha256': sources(),
            'files_sha256': {name: c.sha(DATA / name) for name in sorted(RELEASE_FILES)},
            'duplicate_disclosure_sha256': DISCLOSURE_SHA, 'duplicate_metrics': check_duplicates(episodes),
            'duplicate_handling': 'keep_original_rows_without_deduplication', 'supervision': POLICY,
            'review_kind': KIND, 'review_provenance': PROVENANCE, 'review_sample_pairs': 60, 'review_sample_rows': 120,
            'per_row_human_inspection_attested': False, 'real_tokenizer_audited': True, 'ready_for_training': True,
            'training_authorized': False, 'training_started': False, 'gpu_inference_authorized': False,
            'gpu_inference_started': False, 'model_weights_loaded': False, 'old_final_test_read': False})
        result = validate_release(DATA, gate, require_completion=False)
        p.write(DATA / 'status.json', {'status': 'complete', 'manifest_sha256': c.sha(DATA / 'manifest.json'), 'training_started': False})
        p.write(REPORT / 'verification.json', result)
        p.write(REPORT / 'status.json', {'status': 'complete', 'verification_sha256': c.sha(REPORT / 'verification.json'), 'training_started': False})
        return result
    except BaseException as exc:
        p.text(REPORT / 'failure.log', traceback.format_exc())
        failure = {'status': 'failed', 'error': repr(exc), 'ready_for_training': False, 'training_started': False}
        p.write(REPORT / 'failure.json', failure)
        if DATA.exists(): p.write(DATA / 'failure.json', failure)
        raise
