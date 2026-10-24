from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from extensions.ticket_policy_retention_extension_v1.pipeline import main
if __name__=='__main__': main()
