"""
Streamlit UI for Machine Failure Prediction.
Loads the trained model from the repo.
"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model" / "best_machine_failure_model_v1.joblib"

st.set_page_config(page_title="Machine Failure Prediction", page_icon="⚙️")
st.title("Machine Failure Prediction App")
st.write(
    "This application predicts the likelihood of a machine failing based on its "
    "operational parameters. Enter the sensor and configuration data below to "
    "get a prediction."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found at `{MODEL_PATH}`.\n\n"
        "The GitHub Actions pipeline trains the model and commits it back to the "
        "repo. If you just pushed the repo, wait for the **MLOps Pipeline** "
        "workflow to finish, then reload this page (or restart the Codespace)."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

# Input widgets
Type         = st.selectbox("Machine Type", ["H", "L", "M"])
air_temp     = st.number_input("Air Temperature (K)",      250.0, 400.0, 298.0, 0.1)
process_temp = st.number_input("Process Temperature (K)",  250.0, 500.0, 324.0, 0.1)
rot_speed    = st.number_input("Rotational Speed (RPM)",   0, 3000, 1400)
torque       = st.number_input("Torque (Nm)",              0.0, 100.0, 40.0, 0.1)
tool_wear    = st.number_input("Tool Wear (min)",          0, 300, 10)

# LabelEncoder on sorted ["H","L","M"] gives H=0, L=1, M=2
type_map = {"H": 0, "L": 1, "M": 2}

input_data = pd.DataFrame([{
    "Air temperature":    air_temp,
    "Process temperature": process_temp,
    "Rotational speed":   rot_speed,
    "Torque":             torque,
    "Tool wear":          tool_wear,
    "Type":               type_map[Type],
}])

if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "⚠️ Machine Failure" if prediction == 1 else "✅ No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
