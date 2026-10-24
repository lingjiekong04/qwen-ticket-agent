"""Sanitized offline CPU child. Reject existing outputs; retain all diagnostics."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extensions.ticket_policy_retention_build_v1 import core as c,pipeline as p
from extensions.ticket_policy_audit_v1 import core as audit

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=('remote','compare'),required=True)
    parser.add_argument('--model-path',type=Path,default=Path(audit.s.MODEL)); args=parser.parse_args()
    p.fresh(c.REPORT/args.stage)
    logs=c.REPORT/'logs'; logs.mkdir(parents=True,exist_ok=True)
    path=logs/(args.stage+'.log'); envpath=logs/(args.stage+'-environment.json')
    p.fresh(path); p.fresh(envpath)
    environment,removed=audit.cpu_environment(os.environ); audit.cpu_gate(environment)
    framework=ROOT/'LlamaFactory-ca75f1edf3cb50343ed1c98605141c3e22075b5f/src'
    if args.stage=='remote':
        c.require(sys.version_info[:2]==(3,11),'Remote audit requires fixed Python 3.11')
        c.require(framework.is_dir(),'Pinned LLaMA-Factory source directory missing')
        environment['PYTHONPATH']=str(framework)+os.pathsep+str(ROOT)
    p.write(envpath,{'removed_distributed_keys':removed,'CUDA_VISIBLE_DEVICES':'','offline':True,
                    'python':sys.version.split()[0],'training_started':False,'model_weights_loaded':False})
    command=[sys.executable,'-u','-B','-m','extensions.ticket_policy_retention_build_v1.remote','--stage',args.stage,'--model-path',str(args.model_path)]
    with path.open('x',encoding='utf-8') as stream:
        child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in child.stdout: stream.write(line); stream.flush(); print(line,end='',flush=True)
        code=child.wait()
    c.require(code==0,f'CPU audit failed ({code}); retain log and outputs: {path}')
if __name__=='__main__': main()
