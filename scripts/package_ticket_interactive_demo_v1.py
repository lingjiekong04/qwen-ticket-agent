#!/usr/bin/env python3
"""Freeze a small self-contained demo source package, without weights/holdouts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DIRS=[
    'ticket_runtime_controller_v1_1_design','ticket_runtime_controller_v1_1',
    'ticket_runtime_contract_guard_v1','ticket_runtime_contract_guard_v1_1',
    'ticket_runtime_contract_guard_integration_v1','ticket_runtime_transaction_recovery_v1_2',
    'ticket_runtime_blocked_action_recovery_v1_3','ticket_runtime_blocked_action_recovery_v1_4',
    'ticket_runtime_evidence_completion_v1_5','ticket_runtime_exact_create_recovery_v1_6',
    'ticket_runtime_trusted_completion_v1_7',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    archive=ROOT/'lora_sft_ticket_interactive_demo_v1.zip'
    binding=ROOT/'extensions/ticket_interactive_demo_v1/bindings.json'
    report=ROOT/'reports/ticket-interactive-demo-v1/local-verification.json'
    if archive.exists() or binding.exists() or report.exists():
        raise ValueError('Demo freeze output already exists; preserve and stop')
    files=['extensions/ticket_agent_v1/ticket_env.py']
    for name in DIRS:
        files.extend('extensions/'+name+'/'+leaf for leaf in ('__init__.py','core.py'))
    files.extend(str(p.relative_to(ROOT)).replace('\\','/') for p in sorted(
        (ROOT/'extensions/ticket_interactive_demo_v1').glob('*')) if p.suffix in {'.py','.html'})
    files.extend([
        'configs/ticket_interactive_demo_v1.json','data/ticket_interactive_demo_v1/fixture.json',
        'scripts/run_ticket_interactive_demo_v1.py','scripts/package_ticket_interactive_demo_v1.py',
        'reports/ticket-interactive-demo-v1/handoff.md',
    ])
    files=sorted(set(files))
    save(binding,{'demo_id':'ticket-interactive-demo-v1','runtime':'trusted_completion_v1.7',
                  'files_sha256':{name:sha(ROOT/name) for name in files}})
    checked=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_ticket_interactive_demo_v1.py'),
                            '--stage','smoke'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    print(checked.stderr)
    print(checked.stdout)
    if checked.returncode:
        raise ValueError('Demo smoke verification failed; do not distribute')
    save(report,{'passed':True,'stage':'CPU_interactive_demo_transport_runtime_session_verification',
                 'tests_run':20,'provider':'scripted','runtime':'trusted_completion_v1.7',
                 'model_loaded':False,'gpu_initialized':False,'live_tools_executed':0,
                 'real_model_interactive_validation':'pending_server_startup',
                 'bindings_sha256':sha(binding)})
    files.extend([str(binding.relative_to(ROOT)).replace('\\','/'),str(report.relative_to(ROOT)).replace('\\','/')])
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as output:
        for name in sorted(files):
            info=zipfile.ZipInfo(name,date_time=(2026,10,22,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            output.writestr(info,(ROOT/name).read_bytes())
    print(json.dumps({'archive':str(archive),'sha256':sha(archive),'members':len(files),'passed':True,
                      'local_model_inference_verified':False,'model_weights_in_archive':False},indent=2))


if __name__=='__main__':
    main()
