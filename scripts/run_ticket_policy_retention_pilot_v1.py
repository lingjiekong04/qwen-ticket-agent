"""Only offline pilot build/verify; no bulk/training/GPU options."""
from pathlib import Path
import runpy
import sys

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    runpy.run_module('extensions.ticket_policy_retention_pilot_v1.pipeline',run_name='__main__')
