"""Portable paths for the unchanged, hash-bound demo launcher."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def relocated_config(original, model_dir, adapter_dir):
    result = json.loads(json.dumps(original))
    for key, value in [('model_path', model_dir), ('adapter_path', adapter_dir)]:
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ValueError(key + ' must be a directory')
        result[key] = str(path)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['verify', 'smoke', 'serve'], required=True)
    parser.add_argument('--provider', choices=['scripted', 'local-qwen'], default='local-qwen')
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--adapter-dir', type=Path)
    args = parser.parse_args()
    needs_paths = args.server or (args.stage == 'serve' and args.provider == 'local-qwen')
    if needs_paths and not (args.model_dir and args.adapter_dir):
        parser.error('Provide both --model-dir and --adapter-dir for local model verification/startup')
    from extensions.ticket_interactive_demo_v1 import provider
    from scripts import run_ticket_interactive_demo_v1 as frozen
    provider.verify_sources()
    context = tempfile.TemporaryDirectory(prefix='ticket-demo-config-') if needs_paths else nullcontext()
    with context as temp:
        if needs_paths:
            config = relocated_config(json.loads(provider.CONFIG.read_text(encoding='utf-8')),
                                      args.model_dir, args.adapter_dir)
            temporary_config = Path(temp) / 'config.json'
            temporary_config.write_text(json.dumps(config, ensure_ascii=False), encoding='utf-8')
            provider.CONFIG = temporary_config
        sys.argv = [str(ROOT / 'scripts/run_ticket_interactive_demo_v1.py'),
                    '--stage', args.stage, '--provider', args.provider, '--port', str(args.port)]
        if args.server:
            sys.argv.append('--server')
        frozen.main()

if __name__ == '__main__':
    main()
