"""CPU-only immutable retention data release; default verification is read-only."""
import argparse
import json
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.ticket_policy_audit_v1 import core as audit
from extensions.ticket_policy_retention_release_v1 import core as release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('build', 'verify'), default='verify')
    parser.add_argument('--accept-review')
    parser.add_argument('--data', type=Path, default=release.DATA)
    args = parser.parse_args()
    env, _ = audit.cpu_environment(os.environ)
    os.environ.clear(); os.environ.update(env)
    if args.stage == 'build':
        release.c.require(args.data == release.DATA, 'Builder uses fixed exclusive new directory')
        result = release.build(args.accept_review)
    else:
        release.c.require(args.accept_review is None, 'Verifier cannot record acceptance')
        result = release.verify(args.data)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
