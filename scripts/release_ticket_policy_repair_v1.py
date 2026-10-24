"""Build or read-only verify the independent repair data release. Never train."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.ticket_policy_repair_release_v1 import core as release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('build', 'verify'), default='verify')
    parser.add_argument('--accept-review', help='Only the builder records explicit user acceptance')
    parser.add_argument('--data', type=Path, default=release.DATA)
    args = parser.parse_args()
    if args.stage == 'build':
        release.s.require(args.data == release.DATA, 'Builder uses a fixed fresh release directory')
        result = release.build(args.accept_review)
    else:
        release.s.require(args.accept_review is None, 'Verifier cannot record acceptance')
        result = release.verify(args.data)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
