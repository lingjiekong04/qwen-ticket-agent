"""Read-only CPU reproduction of both v6 scores; never generates or writes."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    from extensions.ticket_runtime_prospective_blind_v6_eval import compare
    data = ROOT / 'data/ticket_runtime_prospective_holdout_v6'
    run = ROOT / 'results/v6'
    load = lambda path: json.loads(path.read_text(encoding='utf-8'))
    for name, expected in compare.s.FILES_SHA256.items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == expected, name
    for name, expected in compare.s.RUNTIME_SOURCE_SHA256.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    expected_comparison = '14ef97845d4d9dc9b5e8e74bbb9b47808f20755c4efdc8901dc6c2c5417fb037'
    assert hashlib.sha256((run / 'comparison.json').read_bytes()).hexdigest() == expected_comparison
    frozen = load(run / 'comparison.json')
    tasks, oracles = load(data / 'tasks.json'), load(data / 'oracle.json')
    assert len(tasks) == len(oracles) == len({task['id'] for task in tasks}) == 160
    compare.s.RUN = run
    counts = {}
    for arm in compare.s.ARMS:
        actual, _ = compare.summarize(tasks, oracles, arm)
        assert actual == frozen['arms'][arm], 'Frozen score mismatch: ' + arm
        counts[arm] = actual['completed']
    assert compare.formal_assessment(frozen['arms'][compare.s.ARMS[1]]) == frozen['formal_assessment']
    print(json.dumps({'passed': True, 'episodes_read': 320, 'completed': counts,
                      'comparison_sha256': expected_comparison,
                      'scope': 'Frozen scorer reproduction, not fresh blind test or human review',
                      'model_loaded': False, 'original_results_modified': False}, indent=2))

if __name__ == '__main__':
    main()
