"""Exclusive-create incremental handoff archive; never includes model weights."""
from pathlib import Path
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extensions.ticket_policy_retention_pilot_v1 import core as c, pipeline as p


def package():
    result = p.verify()
    c.require((c.REPORT/'ai-review.json').is_file() and (c.REPORT/'verification.json').is_file(), 'Pilot review/receipt missing')
    receipt = c.read(c.REPORT/'verification.json')
    c.require(receipt['passed'] and receipt['manifest_sha256']==result['manifest_sha256'], 'Receipt is not bound to pilot')
    for name,expected in receipt['review_files_sha256'].items():
        c.require('/' not in name and '\\' not in name and c.sha(c.REPORT/name)==expected, 'Review artifact changed')
    files = set((c.ROOT/name) for name in p.sources())
    for directory in (c.OUT,c.REPORT,c.PLAN.parent):
        files.update(path for path in directory.iterdir() if path.is_file())
    for name in c.read(c.PLAN)['source_evidence']:
        files.add(c.ROOT/name)
    files.update([ROOT/'scripts/verify_ticket_policy_retention_plan_v1.py',ROOT/'tests/test_ticket_policy_retention_plan_v1.py'])
    destination = ROOT/'lora_sft_ticket_policy_retention_pilot_v1.zip'
    c.require(not destination.exists(), 'Archive exists; refuse overwrite')
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            name = path.relative_to(ROOT).as_posix()
            c.require(path.is_file() and '..' not in Path(name).parts, 'Unsafe archive path')
            archive.write(path,name)
    with zipfile.ZipFile(destination) as archive:
        c.require(archive.testzip() is None,'Archive integrity failure')
        for name in archive.namelist():
            c.require(c.sha(ROOT/name)==__import__('hashlib').sha256(archive.read(name)).hexdigest(),'Archive content differs')
    print(json.dumps({'passed':True,'path':str(destination),'sha256':c.sha(destination),'files':len(files),
                      'rows':40,'ready_for_training':False,'sample_acceptance':'pending'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    package()
