# Machine Failure Prediction

Predictive maintenance for a manufacturing line, trained on sensor + operational
data. The whole pipeline lives inside GitHub — no external hosting needed.

## How it works

1. **`week_2_mls/data/machine-failure-prediction.csv`** is the source of truth
   for the dataset (committed to the repo).
2. **GitHub Actions** (`.github/workflows/pipeline.yml`) runs on every push:
   validates the data, splits it, trains an XGBoost model with GridSearchCV,
   and commits `best_machine_failure_model_v1.joblib` back to the repo.
3. **GitHub Codespaces** (`.devcontainer/devcontainer.json`) auto-launches
   `streamlit run` on port 8501 when you open the repo. The port is
   auto-forwarded and made public.

## Try it

Click **Code → Codespaces → Create codespace on main**. Give it ~30 seconds and
the Streamlit app opens automatically in a new browser tab.
