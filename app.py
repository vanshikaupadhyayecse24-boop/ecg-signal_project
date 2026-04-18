import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load trained model
model = joblib.load("heart_model.pkl")

# Page setup
st.set_page_config(page_title="ECG AI System", layout="wide")

st.title("❤️ AI-Based ECG Disease Detection System")

menu = st.sidebar.selectbox("Menu", ["Home", "Upload ECG", "About"])

# ---------------- HOME ----------------
if menu == "Home":
    st.write("Welcome 👋")
    st.write("Upload ECG signal (CSV file) to detect heart condition.")

# ---------------- UPLOAD ECG ----------------
elif menu == "Upload ECG":
    file = st.file_uploader("Upload ECG CSV File", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)

        st.subheader("📊 Data Preview")
        st.write(data.head())

        # Convert everything to numeric (fix errors)
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.fillna(0)

        # ---------------- ECG GRAPH ----------------
        st.subheader("📈 ECG Waveform Graph")

        fig, ax = plt.subplots()

        # plot first column only
        ax.plot(data.iloc[:, 0].values[:500])

        ax.set_title("ECG Signal Waveform")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")

        st.pyplot(fig)

        # ---------------- FEATURE ENGINEERING ----------------
        features = pd.DataFrame([{
            "mean": data.values.mean(),
            "std": data.values.std(),
            "max": data.values.max(),
            "min": data.values.min()
        }])

        # ---------------- PREDICTION ----------------
        prediction = model.predict(features)[0]

        st.subheader("🧠 Result")

        if prediction == 1:
            st.error("⚠ High Risk of Heart Disease Detected")
        else:
            st.success("✅ Normal ECG Signal")

        st.info("This is an AI prediction, not a medical diagnosis.")

# ---------------- ABOUT ----------------
elif menu == "About":
    st.write("AI system for ECG signal analysis and heart disease prediction.")