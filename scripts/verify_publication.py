"""Verify the allowlisted public tree; print finding locations, never secret values."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'github_token': r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{25,}',
    'openai_token': r'\bsk-(?:proj-)?[A-Za-z0-9_-]{25,}',
    'hf_token': r'\bhf_[A-Za-z0-9]{24,}',
    'cloud_access_id': r'\bAKIA[0-9A-Z]{16}\b',
}

def main():
    manifest_path = ROOT / 'publication/files-sha256.json'
    expected = json.loads(manifest_path.read_text(encoding='utf-8'))['files']
    findings = []
    for rel, digest in expected.items():
        path = ROOT / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            findings.append({'kind': 'hash_mismatch_or_missing', 'path': rel})
            continue
        if path.stat().st_size > 50 * 1024**2 or path.suffix in {'.safetensors', '.pt', '.pth', '.bin', '.zip', '.gz'}:
            findings.append({'kind': 'unwanted_binary_or_large_file', 'path': rel})
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeError:
            continue
        for kind, pattern in PATTERNS.items():
            for match in re.finditer(pattern, text):
                findings.append({'kind': kind, 'path': rel, 'line': text.count('\n', 0, match.start()) + 1})
    # Reject extra files that would be accidentally staged; ignore only runtime artifacts.
    ignored_parts = {'.git', '__pycache__', '.pytest_cache', '.venv', 'runs'}
    for path in ROOT.rglob('*'):
        if not path.is_file() or ignored_parts.intersection(path.relative_to(ROOT).parts):
            continue
        rel = path.relative_to(ROOT).as_posix()
        if path != manifest_path and rel not in expected:
            findings.append({'kind': 'unlisted_file', 'path': rel})
    print(json.dumps({'passed': not findings, 'manifest_files': len(expected), 'findings': findings,
                      'scope': 'Allowlist, hashes and credential-pattern checks; not proof of absence of secrets'}, indent=2))
    if findings:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
