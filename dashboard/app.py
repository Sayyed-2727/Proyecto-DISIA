import streamlit as st
import requests

st.set_page_config(page_title="MIMIC TRIAGE Dashboard", layout="wide")

st.title("🏥 MIMIC-TRIAGE Risk Prediction Dashboard")
st.markdown(
    "Introduce las variables clínicas del paciente para estimar el riesgo de mortalidad hospitalaria."
)

API_URL = "http://mimic-api:8000/predict"
HEALTH_URL = "http://mimic-api:8000/health"

with st.sidebar:
    st.subheader("🧪 API Status")
    try:
        health = requests.get(HEALTH_URL, timeout=2).json()
        if health.get("status") == "ok":
            st.success("API: OK")
        else:
            st.warning(f"API: {health}")
    except Exception as e:
        st.error(f"API: DOWN ({e})")

# -----------------------------
# Sidebar - Clinical Features
# -----------------------------
st.sidebar.header("🩺 Patient Clinical Features")

age = st.sidebar.number_input("Age", min_value=0.0, max_value=120.0, value=65.0)
heart_rate_mean = st.sidebar.number_input("Heart Rate Mean (bpm)", value=85.0)
sysbp_mean = st.sidebar.number_input("Systolic BP Mean (mmHg)", value=110.0)
diasbp_mean = st.sidebar.number_input("Diastolic BP Mean (mmHg)", value=60.0)
resp_rate_mean = st.sidebar.number_input("Respiratory Rate Mean", value=18.0)
temperature_mean = st.sidebar.number_input("Temperature Mean (°C)", value=37.2)
spo2_mean = st.sidebar.number_input("SpO2 Mean (%)", value=96.0)
glucose_mean = st.sidebar.number_input("Glucose Mean (mg/dL)", value=140.0)

input_data = {
    "age": age,
    "heart_rate_mean": heart_rate_mean,
    "sysbp_mean": sysbp_mean,
    "diasbp_mean": diasbp_mean,
    "resp_rate_mean": resp_rate_mean,
    "temperature_mean": temperature_mean,
    "spo2_mean": spo2_mean,
    "glucose_mean": glucose_mean,
}

# -----------------------------
# Prediction Button
# -----------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Patient Summary")
    st.json(input_data)

with col2:
    st.subheader("Prediction")

    if st.button("Predict Mortality Risk", type="primary"):
        try:
            response = requests.post(API_URL, json=input_data, timeout=10)

            if response.status_code == 200:
                result = response.json()

                risk = result.get("mortality_risk_probability", None)
                high_risk = result.get("high_risk", False)
                model_version = result.get("model_version", "unknown")

                if risk is not None:
                    st.caption(f"Model served: {model_version}")
                    st.metric("Mortality Risk Probability", f"{risk:.2%}")

                    if high_risk:
                        st.error("High Mortality Risk Detected")
                    else:
                        st.success("Low Mortality Risk")
                else:
                    st.error(f"Unexpected response: {result}")
            else:
                st.error(f"API Error: {response.text}")

        except Exception as e:
            st.error(f"Connection Error: {e}")
