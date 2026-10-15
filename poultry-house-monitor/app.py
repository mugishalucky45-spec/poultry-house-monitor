import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Poultry-House Monitor", page_icon="🐔", layout="centered")
st.title("MUGISHA")
st.title("🐔 Poultry-House Monitor")
st.write("Classify poultry-house conditions from temperature, humidity, and light level using K-Nearest Neighbours (KNN).")
st.warning("DEMONSTRATION ONLY: trained on synthetic, rule-generated data. Scores and predictions are not validated poultry-welfare guidance. Replace with real, expert-labelled data before operational use.")

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "poultry_knn_model.joblib")

@st.cache_data
def load_metrics():
    return json.loads((ROOT / "evaluation_metrics.json").read_text(encoding="utf-8"))

model = load_model()
metrics = load_metrics()
st.subheader("Environmental readings")
with st.form("predict"):
    c1,c2 = st.columns(2)
    with c1:
        temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.5)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
    with c2:
        light = st.number_input("Light level (lux)", min_value=0.0, max_value=10000.0, value=250.0, step=10.0)
    submitted = st.form_submit_button("Classify conditions", type="primary")
if submitted:
    sample = pd.DataFrame([{"Temperature_C":temperature,"Humidity_pct":humidity,"Light_Level_lux":light}])
    result = model.predict(sample)[0]
    confidence = float(max(model.predict_proba(sample)[0]))*100
    (st.success if result == "Normal" else st.error)(f"Predicted condition: {result}")
    st.metric("Model confidence (demo)", f"{confidence:.1f}%")
    st.caption("Confidence is a model estimate, not a guarantee of safety.")
    st.dataframe(sample, use_container_width=True)

st.divider()
st.subheader("Model evaluation")
a,b,c = st.columns(3)
a.metric("Test accuracy", f"{metrics['test_accuracy']*100:.1f}%")
b.metric("Macro precision", f"{metrics['test_precision_macro']*100:.1f}%")
c.metric("Macro F1-score", f"{metrics['test_f1_macro']*100:.1f}%")
st.write(f"Training rows: **{metrics['train_rows']}** · Test rows: **{metrics['test_rows']}**")
st.write(f"Selected KNN settings: **{metrics['best_parameters']}**")
labels = metrics["confusion_matrix_labels"]
cm = pd.DataFrame(metrics["confusion_matrix"], index=[f"Actual {x}" for x in labels], columns=[f"Predicted {x}" for x in labels])
st.write("Confusion matrix")
st.dataframe(cm, use_container_width=True)
with st.expander("About the synthetic dataset"):
    st.write("The CSV contains 1,200 simulated examples. Labels use illustrative ranges and a small amount of random label noise; ranges are not poultry-farming recommendations.")
    st.write("Input columns: Temperature_C, Humidity_pct, Light_Level_lux. Target: Condition.")
    st.download_button("Download synthetic dataset (CSV)", data=(ROOT/"synthetic_poultry_house_data.csv").read_bytes(), file_name="synthetic_poultry_house_data.csv", mime="text/csv")
