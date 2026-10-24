"""Read-only CPU design verification. Does not generate episodes or load models."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'reports/ticket-policy-retention-next-stage-v1/plan.json'
EXPECTED_QUOTAS = {'knowledge_chain': 5, 'authority': 4, 'priority': 4,
                   'clarification_continuation': 2, 'retention': 5}
EVIDENCE = {
    'reports/ticket-policy-repair-training-evidence-review-v1/report.md': '69a07a6e285c6a4b72b7a87a4d02c16cdc1e717bba271cda10a55ad905f411bd',
    'reports/ticket-policy-repair-training-evidence-review-v1/review.json': '426ff887ee70698d9d7a506eac4f6b6d696a34c65c91006ac9ae1c57ff0aa40f',
    'reports/ticket-policy-repair-training-evidence-review-v1/status.json': 'f373616c33931169ca32a168a2d9d4b9e050c3884026f5d3d322a29117d01396',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(plan):
    require(plan['version'] == 'ticket-policy-retention-next-stage-v1', 'Wrong version')
    require(plan['stage'] == 'design_only', 'Not design-only')
    require(plan['seed'] == 2026100801, 'Seed changed')
    require(plan['authorization'] == {'plan_preparation': True, 'representative_generation': False,
            'bulk_generation': False, 'training': False, 'gpu_inference': False}, 'Authority expanded')
    require(plan['review'] == {'type': 'AI_assisted_design', 'human_approval': 'pending',
                              'per_row_human_inspection': False}, 'Approval misrepresented')
    require(plan['source_evidence'] == EVIDENCE, 'Evidence binding changed')
    boundary = plan['boundaries']
    require(boundary == {'legacy_artifacts_immutable': True, 'dev_rows': 60,
            'dev_permanently_development_only': True, 'dev_in_training': False,
            'existing_diagnostic_rows': 72, 'existing_diagnostics_are_held_out': False,
            'old_final_test_read': False, 'model_visible_hidden_metadata': False,
            'batch_rows': None, 'training_recipe': None, 'training_start_adapter': None}, 'Boundary changed')
    pilot = plan['pilot']
    require((pilot['pairs'], pilot['episodes'], pilot['locales']) == (20, 40, ['zh', 'en']), 'Wrong pilot size')
    require(pilot['status'] == 'specifications_only_not_generated_episodes', 'Specifications are not episodes')
    cases = pilot['cases']
    require(len(cases) == 20 and {c['id'] for c in cases} == {f'nr1-p{i:02}' for i in range(1, 21)}, 'Case IDs/count invalid')
    require(dict(Counter(c['theme'] for c in cases)) == EXPECTED_QUOTAS == pilot['quotas_pairs'], 'Coverage quota changed')
    profiles = {'sentence', 'json', 'explicit_actions'}
    require(set(pilot['permission_profiles']) == profiles, 'Permission profiles changed')
    for case in cases:
        require(case['permission_profile'] in profiles and type(case['allow_write']) is bool, 'Missing/invalid policy')
        require(bool(case['requirement'].strip()), 'Missing supervision requirement')
        require(not ({'row', 'events', 'conversations', 'tool_results', 'final_state'} & set(case)), 'Generated payload in design')
    by_id = {c['id']: c for c in cases}
    require(not by_id['nr1-p06']['allow_write'] and by_id['nr1-p09']['allow_write'], 'Counterfactual permission invalid')
    require(by_id['nr1-p06']['contrast_group'] == by_id['nr1-p09']['contrast_group'], 'Counterfactual group missing')
    denied = [c for c in cases if c['theme'] == 'authority' and not c['allow_write']]
    require({c['permission_profile'] for c in denied} == profiles, 'Denied formats not covered')
    require(all(c['allow_write'] for c in cases if c['theme'] == 'clarification_continuation'), 'Continuation permission invalid')
    require(len(plan['gates']) == 6 and len(plan['future_acceptance']) == 8, 'Missing review/acceptance gates')
    return {'passed': True, 'scope': 'Design structure and provenance only; not episode/tokenizer/model validation',
            'proposed_pairs': 20, 'proposed_episodes': 40, 'quotas_pairs': EXPECTED_QUOTAS,
            'human_approval': 'pending', 'episodes_generated': 0, 'training_started': False,
            'gpu_inference_started': False, 'old_final_test_read': False}


def inspect(root=ROOT):
    root = Path(root)
    plan_path = root / PLAN
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    result = validate(plan)
    for name, expected in EVIDENCE.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        require(actual == expected, 'Evidence hash mismatch: ' + name)
    # Existing verifier is a read-only standard-library dependency hash check.
    from scripts.verify_ticket_policy_repair_training_dependencies_v1 import inspect as legacy_inspect
    legacy = legacy_inspect(root)
    require(legacy['passed'], 'Legacy sources/dependencies changed: ' + str(legacy['issues']))
    result.update({'evidence_files_checked': len(EVIDENCE),
                   'legacy_bound_files_checked': legacy['checked_source_and_dependency_files'],
                   'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest()})
    return result


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    try:
        result = inspect()
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        result = {'passed': False, 'error': f'{type(error).__name__}: {error}', 'training_started': False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
