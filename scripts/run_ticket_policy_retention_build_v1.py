"""Offline only, append-only build stages. No training or model inference."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extensions.ticket_policy_retention_build_v1.pipeline import main
if __name__=='__main__': main()
