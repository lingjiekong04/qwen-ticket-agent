"""Exclusive-create incremental ZIP; prerequisites remain unchanged."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extensions.ticket_policy_retention_extension_v1 import core as c,pipeline as p


def package():
    result=p.verify()
    receipt=c.read(c.REPORT/'verification.json')
    c.require(receipt['passed'] and receipt['manifest_sha256']==result['manifest_sha256'],'Review receipt unbound')
    for name,expected in receipt['review_files_sha256'].items(): c.require(c.sha(c.REPORT/name)==expected,'Review changed')
    files={ROOT/name for name in p.sources()}
    for directory in (c.OUT,c.REPORT,c.DESIGN): files.update(x for x in directory.iterdir() if x.is_file())
    files.update({ROOT/'scripts/freeze_ticket_policy_retention_design_v1.py',ROOT/'tests/test_ticket_policy_retention_design_v1.py'})
    destination=ROOT/'lora_sft_ticket_policy_retention_extension_v1.zip'
    c.require(not destination.exists(),'Archive exists; refuse overwrite')
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            c.require(path.is_file() and path.resolve().is_relative_to(ROOT.resolve()),'Unsafe/missing archive path')
            archive.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(destination) as archive:
        c.require(archive.testzip() is None,'Archive corrupted')
        for name in archive.namelist(): c.require(hashlib.sha256(archive.read(name)).hexdigest()==c.sha(ROOT/name),'Archive member differs')
    report={'passed':True,'archive':destination.name,'archive_sha256':c.sha(destination),'files':len(files),
            'rows':24,'pairs':12,'manifest_sha256':result['manifest_sha256'],'human_acceptance':'pending',
            'ready_for_training':False,'training_started':False,'gpu_inference_started':False,
            'prerequisites':'Existing repository and accepted retention-pilot-v1 package, frozen old simulation/training/DEV sources. No old files overwritten.'}
    p.write_new(c.REPORT/'delivery.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': package()
