#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.ticket_interactive_demo_v1 import core, provider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['verify', 'smoke', 'serve'], required=True)
    parser.add_argument('--server', action='store_true', help='CPU hash/stack verification of server assets')
    parser.add_argument('--provider', choices=['scripted', 'local-qwen'], default='local-qwen')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    checked = provider.verify_sources()
    if args.server:
        checked['server_assets'] = provider.verify_server()
    if args.stage == 'verify':
        print(json.dumps(checked, ensure_ascii=False, indent=2))
    elif args.stage == 'smoke':
        suite = unittest.defaultTestLoader.loadTestsFromName('extensions.ticket_interactive_demo_v1.tests')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        print(json.dumps({'passed':True, 'tests_run':result.testsRun, 'model_loaded':False,
                          'gpu_initialized':False, 'live_tools_executed':0}, indent=2))
    else:
        core.require(1024 <= args.port <= 65535, 'Use an unprivileged valid port')
        # Bind first, so an occupied port fails before a model consumes GPU memory.
        from extensions.ticket_interactive_demo_v1.server import create_server
        service = core.DemoService(None, provider_kind=args.provider,
                                   audit_dir=ROOT / 'runs/ticket-interactive-demo-v1')
        httpd = create_server(service, port=args.port)
        try:
            print('Loading provider: ' + args.provider, flush=True)
            service.provider = (provider.load_local_provider() if args.provider == 'local-qwen'
                                else core.SyntheticProvider())
            print(json.dumps({'ready':True,'url':f'http://127.0.0.1:{args.port}',
                              'provider':args.provider,'audit_file':str(service.audit_path)},ensure_ascii=False),flush=True)
            httpd.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            print('Demo stopped; in-memory sessions released.',flush=True)
        finally:
            httpd.server_close()


if __name__ == '__main__':
    main()
