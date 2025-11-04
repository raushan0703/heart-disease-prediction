import streamlit as st
import numpy as np
import pickle
import time

# -------------------------- PAGE CONFIG -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# -------------------------- LOAD MODEL & SCALER ---------------------
model = pickle.load(open('heart_disease_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# -------------------------- HEADER SECTION --------------------------
st.markdown("<h1 style='text-align: center; color: red;'>❤️ Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Predict your heart condition based on your health details.</p>", unsafe_allow_html=True)
st.divider()

# -------------------------- INPUT SECTION ---------------------------
st.subheader("🩺 Enter Your Health Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age (in years)', 1, 120, 30)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', 80, 200, 120)
    chol = st.number_input('Serum Cholesterol (mg/dl)', 100, 600, 200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    restecg = st.selectbox('Resting ECG Results (0-2)', [0, 1, 2])

with col2:
    thalach = st.number_input('Maximum Heart Rate Achieved', 60, 220, 150)
    exang = st.selectbox('Exercise Induced Angina (0=No,1=Yes)', [0, 1])
    oldpeak = st.number_input('ST Depression Induced by Exercise', 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox('Slope of ST Segment (0-2)', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels (0-3)', [0, 1, 2, 3])
    thal = st.selectbox('Thal (0=Normal,1=Fixed,2=Reversible)', [0, 1, 2])

# -------------------------- PREDICTION BUTTON -----------------------
st.divider()
st.markdown("<h3 style='color:#00BFFF;'>🔍 Prediction Section</h3>", unsafe_allow_html=True)

if st.button('🔍 Predict'):
    sex = 1 if sex == 'Male' else 0

    # Convert input to array
    user_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal]])
    
    # Scale the input data for better accuracy
    user_data_scaled = scaler.transform(user_data)

    # Animation during prediction
    with st.spinner('Analyzing your heart condition... ⏳'):
        time.sleep(2)
        prediction = model.predict(user_data_scaled)

    # Display result
    st.subheader("🩸 Prediction Result:")
    if prediction[0] == 1:
        st.error('💔 **Heart Disease Detected!** Please consult a cardiologist as soon as possible.')
    else:
        st.success('💚 **No Heart Disease Detected!** You seem to have a healthy heart. Keep it up!')

st.divider()

# -------------------------- INFORMATION SECTION ---------------------
st.subheader("📋 Health Parameter Information and Normal Ranges")

st.markdown("""
| Feature | Description | Normal Range / Meaning |
|----------|--------------|------------------------|
| **Age** | Age of the person | 20 - 70 years |
| **Sex** | Gender of patient | 1 = Male, 0 = Female |
| **cp (Chest Pain Type)** | 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal, 3 = Asymptomatic | 0–3 |
| **trestbps (Resting BP)** | Resting Blood Pressure | 90–140 mm Hg |
| **chol (Cholesterol)** | Serum Cholesterol | 150–250 mg/dl |
| **fbs (Fasting Sugar)** | Fasting Blood Sugar > 120 mg/dl | 0 = No, 1 = Yes |
| **restecg (ECG Results)** | 0 = Normal, 1 = ST-T Abnormality, 2 = LV Hypertrophy | 0–2 |
| **thalach (Max Heart Rate)** | Max heart rate achieved during test | 100–200 bpm |
| **exang (Exercise Angina)** | Pain during exercise | 0 = No, 1 = Yes |
| **oldpeak (ST Depression)** | Depression induced by exercise | 0–2.5 (Normal) |
| **slope (ST Slope)** | Slope of peak exercise ST | 0 = Upsloping, 1 = Flat, 2 = Downsloping |
| **ca (Major Vessels)** | Number of major vessels colored by fluoroscopy | 0–3 |
| **thal** | Thalassemia test result | 0 = Normal, 1 = Fixed Defect, 2 = Reversible |
""")

# -------------------------- FOOTER SECTION --------------------------
st.divider()
st.markdown("""
<div style='text-align:center'>
    <h4>💡 Tips for a Healthy Heart:</h4>
    <ul style='text-align:left; display:inline-block;'>
        <li>🥗 Eat balanced, low-cholesterol food regularly</li>
        <li>🚶‍♂️ Exercise or walk at least 30 minutes daily</li>
        <li>🚭 Avoid smoking and alcohol completely</li>
        <li>🧘‍♀️ Practice meditation or yoga for stress relief</li>
        <li>💧 Drink plenty of water and get enough sleep</li>
    </ul>
    <br>
    <p style='font-size:13px; color:gray;'>Developed by <b>Raushan Jaiswal</b> | B.Tech CSE | 2025</p>
</div>
""", unsafe_allow_html=True)
