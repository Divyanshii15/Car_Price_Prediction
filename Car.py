import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# ---------------- CLEAN BACKGROUND CSS FIX ----------------
st.markdown(
"""
<style>

/* FULL BACKGROUND IMAGE */
.stApp {
    background: url("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7") no-repeat center center fixed;
    background-size: cover;
}

/* DARK OVERLAY FIX (NO TEXT MIX ISSUE NOW) */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.55);
    z-index: 0;
}

/* MAIN CONTENT ABOVE OVERLAY */
.block-container {
    position: relative;
    z-index: 2;
    padding: 2rem;
}

/* TITLE */
h1, h2, h3, h4 {
    color: #ffffff !important;
    text-align: center;
    font-weight: 800;
}

/* SIDEBAR FIX */
section[data-testid="stSidebar"] {
    background: rgba(10, 10, 10, 0.95) !important;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* INPUT FIELDS FIX */
input, textarea {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 8px !important;
}

/* STREAMLIT INPUT BOX FIX */
div[data-baseweb="input"] input {
    background-color: #111827 !important;
    color: white !important;
}

/* SELECT BOX FIX */
div[data-baseweb="select"] > div {
    background-color: #111827 !important;
    color: white !important;
}

/* BUTTON STYLE */
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1rem;
}

/* RESULT BOX (GLASS EFFECT) */
.result-box {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 15px;
    color: #ffffff;
    font-weight: 600;
    text-align: center;
    margin-top: 20px;
}

/* REMOVE EXTRA WHITE FLASH */
.css-1d391kg {
    background: transparent !important;
}

</style>
""",
unsafe_allow_html=True
)

# ---------------- TITLE ----------------
st.title("🚗 Car Price Prediction Dashboard")

# ---------------- LOAD DATA ----------------
file_path = os.path.join(os.getcwd(), "carprice.csv")
data = pd.read_csv(file_path)

# ---------------- CLEANING ----------------
data = data.replace("?", np.nan)
data.dropna(inplace=True)

data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data['price'] = pd.to_numeric(data['price'], errors='coerce')
data['city-mpg'] = pd.to_numeric(data['city-mpg'], errors='coerce')
data['highway-mpg'] = pd.to_numeric(data['highway-mpg'], errors='coerce')

data.dropna(inplace=True)

# ---------------- ENCODING ----------------
data['fuel-type'] = data['fuel-type'].map({'gas': 0, 'diesel': 1})
data['engine-type'] = data['engine-type'].astype('category').cat.codes

# ---------------- FEATURES ----------------
X = data[['horsepower', 'city-mpg', 'highway-mpg', 'engine-type', 'fuel-type']]
y = data['price']

# ---------------- TRAIN MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("🚘 Enter Car Details")

hp = st.sidebar.number_input("Horsepower", 50, 300, 100)
city = st.sidebar.number_input("City MPG", 10, 60, 25)
highway = st.sidebar.number_input("Highway MPG", 10, 70, 30)

engine = st.sidebar.selectbox("Engine Type", sorted(data['engine-type'].unique()))
fuel = st.sidebar.selectbox("Fuel Type", ["gas", "diesel"])

fuel_map = {"gas": 0, "diesel": 1}

# ---------------- PREDICTION ----------------
st.markdown("### 🔮 Prediction Result")

if st.button("🚀 Predict Price"):

    prediction = model.predict([[hp, city, highway, engine, fuel_map[fuel]]])

    st.markdown(
        f"""
        <div class="result-box">
            💰 Predicted Price: ₹ {prediction[0]:,.2f} <br><br>
            📊 Model Accuracy: {accuracy*100:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- FOOTER ----------------
st.caption("🚗 AI Powered Car Price Prediction System | Streamlit Project")