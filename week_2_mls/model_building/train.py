"""
Train an XGBoost model with GridSearchCV, then save the best model to
week_2_mls/deployment/model/best_machine_failure_model_v1.joblib so the
Streamlit app (and any Codespace) can load it directly from the repo.
"""
from pathlib import Path
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

DATA_DIR = Path("week_2_mls/data/processed")
MODEL_DIR = Path("week_2_mls/deployment/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "best_machine_failure_model_v1.joblib"

Xtrain = pd.read_csv(DATA_DIR / "Xtrain.csv")
Xtest = pd.read_csv(DATA_DIR / "Xtest.csv")
ytrain = pd.read_csv(DATA_DIR / "ytrain.csv").squeeze()
ytest = pd.read_csv(DATA_DIR / "ytest.csv").squeeze()

numeric_features = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]
categorical_features = ["Type"]

# Class weight to handle imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__colsample_bytree": [0.4, 0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.5, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

grid_search = GridSearchCV(
    model_pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1
)
grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

print("\nTraining Classification Report:")
print(classification_report(ytrain, best_model.predict(Xtrain)))

print("\nTest Classification Report:")
print(classification_report(ytest, best_model.predict(Xtest)))

joblib.dump(best_model, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
