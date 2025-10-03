import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.data_metadata import update_metadata
import glob

files = glob.glob(str(ROOT / 'data' / 'EURUSD_*.csv')) + glob.glob(str(ROOT / 'data' / 'XAUUSD_*.csv'))
for f in sorted(files):
    ok = update_metadata(f)
    print(f, 'updated' if ok else 'failed')
