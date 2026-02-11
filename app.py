import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px


# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Myo-Sim Bio-Deck", page_icon="🧬", layout="wide")


# --- CUSTOM CSS (Dark Sci-Fi Theme) ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    div.stButton > button:first-child { background-color: #00adb5; color: white; border: none; }
    div.stButton > button:hover { background-color: #007d85; }
    .sidebar .sidebar-content { background-color: #161a23; }
    .stTabs [data-baseweb="tab-list"] { background: #161a23; }
    </style>
    """, unsafe_allow_html=True)


# --- LOAD THE BRAIN ---
@st.cache_resource
def load_model():
    try:
        return joblib.load("myocore_pipeline.pkl")
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

model = load_model()


# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/ios-filled/100/00adb5/heart-with-pulse.png", width=80)
st.sidebar.title("Myo-Sim Bio-Deck")
st.sidebar.markdown("**Chronos Time-Travel Interface**\n\n*CVD Risk Projection Engine*")
st.sidebar.markdown("---")
st.sidebar.info("Adjust patient vitals and run the simulation to see risk projections and simulated outcomes.")

# --- HEADER ---
st.markdown("<h1 style='text-align:center; color:#00adb5;'>Myo-Sim Bio-Deck</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Chronos Time-Travel Interface | <i>CVD Risk Projection Engine</i></p>", unsafe_allow_html=True)
st.markdown("---")


# --- MAIN LAYOUT WITH TABS ---
tab1, tab2, tab3 = st.tabs(["📝 Patient Simulator", "📊 Risk Graphs", "ℹ️ About"])

with tab1:
    st.subheader("Patient Vitals")
    col1, col2 = st.columns([1, 1])
    with col1:
        age = st.slider("Age (Years)", 20, 90, 45)
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trtbps = st.slider("Resting BP (mmHg)", 90, 200, 130)
        chol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
    with col2:
        thalachh = st.slider("Max Heart Rate", 60, 220, 150)
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
        exng = st.selectbox("Exercise Angina?", ["No", "Yes"])
        caa = st.slider("Major Vessels (0-3)", 0, 3, 0)
        thall = st.slider("Thal Rate (0-3)", 0, 3, 2)

    # Convert Inputs to DataFrame
    sex_val = 1 if sex == "Male" else 0
    exng_val = 1 if exng == "Yes" else 0
    cp_val = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[cp]


    # Build input_data with all possible features
    input_dict = {
        'age': [age], 'sex': [sex_val], 'cp': [cp_val], 'trtbps': [trtbps],
        'chol': [chol], 'fbs': [0], 'restecg': [1], 'thalachh': [thalachh],
        'exng': [exng_val], 'oldpeak': [oldpeak], 'slp': [1], 'caa': [caa], 'thall': [thall]
    }
    input_data = pd.DataFrame(input_dict)
    # Align columns to model's expected features, fill missing with 0, ignore extras
    if model is not None and hasattr(model, 'feature_names_in_'):
        for col in model.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0  # Default value for missing features
        input_data = input_data[model.feature_names_in_]

    if st.button("🚀 RUN SIMULATION"):
        if model:
            # Predict
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1] * 100

            # Display Big Gauge
            st.markdown(f"<h1 style='text-align: center; color: #00adb5; font-size: 80px;'>{probability:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>HEART ATTACK PROBABILITY</p>", unsafe_allow_html=True)

            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta", value = probability,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Gauge"},
                gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#00adb5"},
                         'steps': [
                             {'range': [0, 30], 'color': "#1a3c40"},
                             {'range': [30, 70], 'color': "#393e46"},
                             {'range': [70, 100], 'color': "#b23c3c"}
                         ]}
            ))
            fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig, use_container_width=True)

            # Show input summary
            st.markdown("---")
            st.markdown("#### Patient Input Summary")
            st.dataframe(input_data, use_container_width=True)
        else:
            st.error("⚠️ Model file (myocore_pipeline.pkl) not found. Please upload it to GitHub!")

with tab2:
    st.subheader("Risk Graphs & Simulations")
    st.markdown("Visualize how changing each vital affects risk.")
    if model:
        # Simulate risk for age
        ages = list(range(20, 91, 5))
        risks = []
        for a in ages:
            test_data = input_data.copy()
            test_data['age'] = a
            risks.append(model.predict_proba(test_data)[0][1] * 100)
        fig_age = px.line(x=ages, y=risks, labels={'x': 'Age', 'y': 'Risk (%)'}, title='Risk vs Age', markers=True)
        fig_age.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
        st.plotly_chart(fig_age, use_container_width=True)
        # Simulate risk for cholesterol
        chols = list(range(100, 401, 20))
        risks_chol = []
        for c in chols:
            test_data = input_data.copy()
            test_data['chol'] = c
            risks_chol.append(model.predict_proba(test_data)[0][1] * 100)
        fig_chol = px.line(x=chols, y=risks_chol, labels={'x': 'Cholesterol', 'y': 'Risk (%)'}, title='Risk vs Cholesterol', markers=True)
        fig_chol.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
        st.plotly_chart(fig_chol, use_container_width=True)
    else:
        st.info("Graphs will appear after model loads and simulation is run.")

with tab3:
    st.subheader("About Myo-Sim Bio-Deck")
    st.markdown("""
    **Myo-Sim Bio-Deck** is a professional patient simulator for cardiovascular risk projection.\
    Adjust patient vitals, run simulations, and visualize risk using interactive graphs.\
    Built with Streamlit, Plotly, and machine learning.\
    <br><br>
    <b>Author:</b> Your Name<br>
    <b>Version:</b> 1.1 (2026)
    """, unsafe_allow_html=True)
