"""Read-only CPU verification of the unchanged launcher and supplement dependency."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = 'reports/ticket-policy-repair-training-v1/delivery.json'
DELIVERY_SHA = '806101ecaf8cf9da8978102e02c0c12575f0bf90f1dd22b40af6b7de3c9b9fd3'
MISSING = 'reports/ticket-sft-v3-dev-ai-review-20261007.json'
MISSING_SHA = '26cf7ba210976b334abf1685e96a5bf2ec5fa9ed93b5cd0b2e3f9408312729a4'
ORIGINAL_ARCHIVE_SHA = '304ea440a8490e7c887b5db97c117502aaf7f4a480b0578427e6ee9012a6d19e'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def safe_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe manifest path: ' + name)
    resolved = (Path(root) / relative).resolve()
    if not resolved.is_relative_to(Path(root).resolve()): raise ValueError('Dependency escapes project: ' + name)
    return resolved


def inspect(root=ROOT):
    path = safe_path(root, DELIVERY)
    if not path.is_file(): raise ValueError('Original delivery manifest missing: ' + DELIVERY)
    if sha(path) != DELIVERY_SHA: raise ValueError('Original delivery manifest changed; do not re-freeze hashes')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['dependencies'].get(MISSING) != MISSING_SHA:
        raise ValueError('Supplement is not bound by the original delivery')
    hashes = {**manifest['sources'], **manifest['dependencies']}
    hashes.update({'reports/ticket-policy-repair-training-v1/' + name: value
                   for name, value in manifest['report_files_sha256'].items()})
    issues = []
    for name, expected in sorted(hashes.items()):
        dependency = safe_path(root, name)
        if not dependency.is_file(): issues.append({'path': name, 'error': 'missing', 'expected_sha256': expected})
        else:
            actual = sha(dependency)
            if actual != expected: issues.append({'path': name, 'error': 'hash_mismatch', 'expected_sha256': expected, 'actual_sha256': actual})
    status_path = safe_path(root, 'reports/ticket-policy-repair-training-v1/status.json')
    if not status_path.is_file(): issues.append({'path': str(status_path.relative_to(root)), 'error': 'missing'})
    else:
        status = json.loads(status_path.read_text(encoding='utf-8'))
        if status.get('status') != 'complete' or status.get('delivery_sha256') != DELIVERY_SHA:
            issues.append({'path': str(status_path.relative_to(root)), 'error': 'incomplete_original_delivery'})
    return {'passed': not issues, 'checked_source_and_dependency_files': len(hashes), 'issues': issues,
            'restored_report_path': MISSING, 'restored_report_sha256': MISSING_SHA,
            'original_delivery_sha256': DELIVERY_SHA,
            'source_and_training_configuration_changed': False, 'training_started': False,
            'gpu_inference_started': False, 'old_final_test_read': False,
            'scope': 'Hash completeness only; rerun the original --stage verify for all data/replay/DEV gates'}


if __name__ == '__main__':
    try: result = inspect()
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        result = {'passed': False, 'error': f'{type(error).__name__}: {error}', 'training_started': False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
