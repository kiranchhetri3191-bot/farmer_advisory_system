# =========================================================
# FARMER LOAN RISK PREDICTOR SYSTEM
# Decision Support • Climate + Crop + Income Risk
# LEGALLY SAFE • OPEN SOURCE • STREAMLIT READY
# =========================================================

import os
import requests
import joblib
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(
    page_title="Farmer Loan Risk Predictor",
    page_icon="🌾",
    layout="wide"
)

MODEL_FILE = "loan_risk_model.pkl"
TARGET = "loan_risk"

# =========================================================
# LEGAL DISCLAIMER (GLOBAL)
# =========================================================
LEGAL_DISCLAIMER = """
⚠️ **Disclaimer**

This application is a **decision-support and educational tool only**.
It does **NOT** provide financial, legal, or lending advice.

Loan approval, rejection, interest rate, and disbursement decisions
must be made independently by authorized banks or NBFCs after
proper due diligence.

The developers are **NOT responsible** for decisions taken based on
this tool.

Climate and location data are sourced from **publicly available
open-source APIs**.
"""

# =========================================================
# OPEN-SOURCE WEATHER (SAFE & LEGAL)
# =========================================================
def get_lat_lon(city, state):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": f"{city}, {state}, India", "count": 1}
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    if "results" in data:
        return data["results"][0]["latitude"], data["results"][0]["longitude"]
    return None, None


def get_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation",
    }
    r = requests.get(url, params=params, timeout=10)
    return r.json()["current"]

# =========================================================
# DATA LOADER (CSV / EXCEL / DEFAULT SAFE DATA)
# =========================================================
def load_data(file=None):
    if file:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)

    # SAFE SAMPLE DATA (NO REAL PERSON DATA)
    return pd.DataFrame({
        "state": ["Andhra Pradesh", "Karnataka", "Tamil Nadu", "Maharashtra"],
        "crop": ["Rice", "Maize", "Sugarcane", "Cotton"],
        "land_acres": [3, 2, 5, 4],
        "annual_income": [180000, 150000, 250000, 220000],
        "loan_amount": [100000, 80000, 150000, 120000],
        "rainfall": [900, 750, 700, 850],
        "temperature": [32, 28, 33, 30],
        "loan_risk": ["High", "Medium", "Low", "Medium"]
    })

# =========================================================
# UI HEADER
# =========================================================
st.title("🌾 Farmer Loan Risk Predictor")
st.caption("Credit Risk • Climate Risk • Crop Risk (Decision Support Only)")

st.warning(LEGAL_DISCLAIMER)

# =========================================================
# SIDEBAR INPUTS
# =========================================================
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Farmer Loan Data (CSV / Excel)",
    type=["csv", "xlsx"]
)

state = st.sidebar.selectbox(
    "State",
    [
        "Andhra Pradesh", "Karnataka", "Tamil Nadu", "Maharashtra",
        "West Bengal", "Bihar", "Uttar Pradesh"
    ]
)

city = st.sidebar.text_input("Village / Town / City")

# =========================================================
# LOAD DATA
# =========================================================
df = load_data(uploaded_file)

st.subheader("📄 Loan Dataset (Preview)")
st.dataframe(df)

# =========================================================
# WEATHER INFORMATION (ADVISORY ONLY)
# =========================================================
if city:
    lat, lon = get_lat_lon(city, state)
    if lat:
        weather = get_weather(lat, lon)
        st.info(
            f"🌦 **{city} Weather (Advisory)** → "
            f"Temperature: {weather['temperature_2m']}°C | "
            f"Rainfall: {weather['precipitation']} mm"
        )
    else:
        st.warning("Location not found in open database")

# =========================================================
# MACHINE LEARNING PIPELINE (SAFE)
# =========================================================
X = df.drop(columns=[TARGET], errors="ignore")
y = df[TARGET] if TARGET in df.columns else None

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols),
])

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# =========================================================
# TRAIN MODEL (ONLY IF TARGET EXISTS)
# =========================================================
if y is not None:
    if not os.path.exists(MODEL_FILE):
        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_FILE)
        st.success("✅ Loan risk model trained (safe & local)")
    else:
        pipeline = joblib.load(MODEL_FILE)

# =========================================================
# PREDICTION (RISK INDICATOR ONLY)
# =========================================================
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    df["predicted_loan_risk"] = model.predict(X)

    st.subheader("🏦 Loan Risk Indicator")
    st.dataframe(df)

    st.markdown("### 📊 Risk Distribution")
    st.bar_chart(df["predicted_loan_risk"].value_counts())

# =========================================================
# DOWNLOAD (USER CONTROLLED)
# =========================================================
st.download_button(
    "⬇️ Download Risk Report (CSV)",
    df.to_csv(index=False),
    "farmer_loan_risk_report.csv",
    "text/csv"
)

# =========================================================
# TERMS & PRIVACY (LEGAL SAFETY)
# =========================================================
with st.expander("📜 Terms of Use & Privacy Policy"):
    st.markdown("""
**Terms of Use**
- This tool is for educational and decision-support purposes only
- It does not guarantee loan approval or rejection
- Users are responsible for verifying outputs independently

**Privacy Policy**
- No personal data is stored
- No Aadhaar, PAN, phone, or identity data is collected
- Uploaded files are processed in-session only
- No data is shared with third parties
""")

st.caption("© 2025 • Open-source • Educational • Decision Support Tool")
