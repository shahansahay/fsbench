"""Write data/manifest.csv: checksums, sizes, class counts and duplicates of all 18 datasets."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from fsb.data import manifest

out = Path("data/manifest.csv")
m = manifest()
m.to_csv(out, index=False)
pd.set_option("display.width", 200)
print(m.drop(columns=["sha256", "file"]).to_string(index=False))
bad = m[(m.n - m.published_n).astype(bool) | (m.d - m.published_d).astype(bool)]
print("\nDiffers from the published table:", ", ".join(bad.dataset) if len(bad) else "none")
print("Wrote", out)
