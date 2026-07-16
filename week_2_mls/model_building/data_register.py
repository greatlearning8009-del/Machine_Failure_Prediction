"""
Validate the raw dataset that ships with the repo.

In the HuggingFace version this uploaded the CSV to a HF dataset repo.
Now the CSV lives in the repo itself, so we just sanity-check it so the
pipeline fails fast if the file is missing or the schema drifted.
"""
from pathlib import Path
import sys
import pandas as pd

RAW_CSV = Path("week_2_mls/data/machine-failure-prediction.csv")
REQUIRED_COLS = {
    "UDI", "Type", "Air temperature", "Process temperature",
    "Rotational speed", "Torque", "Tool wear", "Failure",
}

if not RAW_CSV.exists():
    sys.exit(f"ERROR: {RAW_CSV} not found. Commit the CSV to the repo first.")

df = pd.read_csv(RAW_CSV)
missing = REQUIRED_COLS - set(df.columns)
if missing:
    sys.exit(f"ERROR: dataset is missing columns: {missing}")

print(f"Dataset OK — {len(df):,} rows, {len(df.columns)} columns.")
print(f"Target class balance:\n{df['Failure'].value_counts()}")
