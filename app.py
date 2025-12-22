import streamlit as st
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Farmer Advisory System",
    page_icon="🌾",
    layout="wide"
)

# ---------------- TITLE ----------------
st.markdown(
    """
    <h1 style='text-align:center; color:#2E8B57;'>
    🌾 Smart Farmer Advisory System
    </h1>
    <p style='text-align:center; font-size:18px;'>
    Crop • Weather • Fertilizer • Pest Advisory
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("👨‍🌾 Farmer Details")

soil_type = st.sidebar.selectbox(
    "Soil Type",
    ["Alluvial", "Black", "Red", "Laterite", "Sandy"]
)

season = st.sidebar.selectbox(
    "Season",
    ["Kharif", "Rabi", "Zaid"]
)

rainfall = st.sidebar.slider(
    "Annual Rainfall (mm)",
    200, 2000, 800
)

temperature = st.sidebar.slider(
    "Average Temperature (°C)",
    10, 45, 28
)

crop_issue = st.sidebar.selectbox(
    "Any Crop Issue?",
    ["None", "Pest Attack", "Yellow Leaves", "Low Yield"]
)

st.sidebar.markdown("---")
st.sidebar.info("Developed using Python + Streamlit")

# ---------------- LOGIC FUNCTIONS ----------------
def recommend_crop(soil, season, rain):
    if season == "Kharif":
        if soil in ["Alluvial", "Black"] and rain > 700:
            return "Rice, Maize, Cotton"
        else:
            return "Millets, Pulses"
    elif season == "Rabi":
        if soil in ["Alluvial", "Red"]:
            return "Wheat, Mustard, Barley"
        else:
            return "Gram, Peas"
    else:
        return "Watermelon, Cucumber, Fodder Crops"

def fertilizer_advice(soil):
    if soil == "Black":
        return "Use Nitrogen & Phosphorus based fertilizers"
    elif soil == "Red":
        return "Add Organic Manure + Potassium"
    elif soil == "Alluvial":
        return "Balanced NPK fertilizer recommended"
    else:
        return "Use Compost & Organic Fertilizers"

def pest_advisory(issue):
    if issue == "Pest Attack":
        return "Use Neem oil spray or consult agri officer"
    elif issue == "Yellow Leaves":
        return "Possible Nitrogen deficiency – apply Urea"
    elif issue == "Low Yield":
        return "Check soil health & irrigation schedule"
    else:
        return "No pest issue detected"

def weather_advice(temp, rain):
    if temp > 35:
        return "High temperature – increase irrigation"
    elif rain < 400:
        return "Low rainfall – use drip irrigation"
    else:
        return "Weather conditions are favorable"

# ---------------- ADVISORY OUTPUT ----------------
st.markdown("## 📊 Farmer Advisory Report")

crop = recommend_crop(soil_type, season, rainfall)
fertilizer = fertilizer_advice(soil_type)
pest = pest_advisory(crop_issue)
weather = weather_advice(temperature, rainfall)

data = {
    "Category": [
        "Recommended Crops",
        "Fertilizer Advice",
        "Pest Advisory",
        "Weather Advisory"
    ],
    "Suggestion": [
        crop,
        fertilizer,
        pest,
        weather
    ]
}

df = pd.DataFrame(data)

st.table(df)

# ---------------- SUMMARY CARDS ----------------
st.markdown("## 🌱 Quick Summary")

col1, col2, col3, col4 = st.columns(4)

col1.success(f"🌾 Crops\n\n{crop}")
col2.info(f"🧪 Fertilizer\n\n{fertilizer}")
col3.warning(f"🐛 Pest\n\n{pest}")
col4.success(f"🌦️ Weather\n\n{weather}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>© 2025 Farmer Advisory System | Python Project</p>",
    unsafe_allow_html=True
)
