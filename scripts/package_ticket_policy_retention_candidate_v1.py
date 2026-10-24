"""Incremental archive only; no model weights, old data or final tests collected."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from extensions.ticket_policy_retention_build_v1 import core as c,pipeline as p

def package():
    result=p.verify(); destination=ROOT/'lora_sft_ticket_policy_retention_candidate_v1.zip'; p.fresh(destination)
    files={ROOT/n for n in p.sources()}
    for directory in (c.OUT,c.EVAL,c.REPORT):
        files.update(x for x in directory.iterdir() if x.is_file() and x.name!='delivery.json')
    # Bundle earlier retention-stage evidence/source so the server need not have
    # executed those LOCAL construction stages. No prior train/model files copied.
    from extensions.ticket_policy_retention_pilot_v1 import core as pilot,pipeline as pilot_pipeline
    from extensions.ticket_policy_retention_extension_v1 import core as ext,pipeline as ext_pipeline
    from extensions.ticket_policy_retention_extension_v1_1 import core as revision,pipeline as revision_pipeline
    for module in (pilot_pipeline,ext_pipeline,revision_pipeline): files.update(ROOT/n for n in module.sources())
    for directory in (pilot.OUT,pilot.REPORT,pilot.PLAN.parent,ext.OUT,ext.REPORT,revision.OUT,revision.REPORT,c.DESIGN):
        files.update(x for x in directory.iterdir() if x.is_file())
    files.update(ROOT/n for n in c.read(pilot.PLAN)['source_evidence'])
    for name in ('scripts/verify_ticket_policy_retention_plan_v1.py','tests/test_ticket_policy_retention_plan_v1.py',
                 'scripts/freeze_ticket_policy_retention_design_v1.py','tests/test_ticket_policy_retention_design_v1.py'):
        files.add(ROOT/name)
    # Incremental server prerequisites are explicit and individually hash-bound.
    prerequisite_hashes=p.dependencies()
    receipt={'passed':True,'archive':destination.name,'rows':1800,'pairs':900,'targeted_eval_rows':192,
      'candidate_manifest_sha256':result['candidate_manifest_sha256'],'targeted_eval_manifest_sha256':result['targeted_eval_manifest_sha256'],
      'files_sha256':{f.relative_to(ROOT).as_posix():c.sha(f) for f in sorted(files)},
      'server_prerequisites_sha256':prerequisite_hashes,'real_tokenizer_audit':'pending_remote','sample_review':'pending_post_CPU_audit',
      'ready_for_training':False,'training_started':False,'gpu_inference_started':False,'old_final_test_read':False}
    p.write(c.REPORT/'package-members.json',receipt)
    files.add(c.REPORT/'package-members.json')
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            c.require(f.is_file() and f.resolve().is_relative_to(ROOT.resolve()),'Unsafe archive member')
            z.write(f,f.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(destination) as z:
        c.require(z.testzip() is None,'Archive corruption')
        for n in z.namelist(): c.require(hashlib.sha256(z.read(n)).hexdigest()==c.sha(ROOT/n),'Archive member differs')
    delivery={k:v for k,v in receipt.items() if k not in ('files_sha256','server_prerequisites_sha256')}|{'archive_sha256':c.sha(destination),'files':len(files)}
    p.write(c.REPORT/'delivery.json',delivery)
    print(json.dumps(delivery,ensure_ascii=False,indent=2))
if __name__=='__main__': package()
