"""
Read raw CSV, encode categoricals, split into train/test,
and save to week_2_mls/data/processed/.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RAW_CSV = Path("week_2_mls/data/machine-failure-prediction.csv")
OUT_DIR = Path("week_2_mls/data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_CSV)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=["UDI"], inplace=True)

# Encode the categorical 'Type' column
df["Type"] = LabelEncoder().fit_transform(df["Type"])

# Split features and target
X = df.drop(columns=["Failure"])
y = df["Failure"]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save splits
Xtrain.to_csv(OUT_DIR / "Xtrain.csv", index=False)
Xtest.to_csv(OUT_DIR / "Xtest.csv", index=False)
ytrain.to_csv(OUT_DIR / "ytrain.csv", index=False)
ytest.to_csv(OUT_DIR / "ytest.csv", index=False)

print(f"Wrote train/test splits to {OUT_DIR}/")
