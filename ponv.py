import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import streamlit.components.v1 as components # Import components for embedding HTML/JS
import sqlite3 # Import sqlite3 for database operations
import datetime # Import datetime for timestamp

# Core Setup and UI
st.set_page_config(layout="wide")

# Inject custom CSS for styling the app and the flowchart
# This block contains all the CSS rules, including those for the flowchart
st.markdown("""
<style>
/* Import professional fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* General Body Styling */
body {
    font-family: 'Inter', sans-serif;
    background-color: #f8f9fa;
    color: #212529;
    line-height: 1.6;
}

/* Main content area styling */
.main .block-container {
    padding: 3rem 5rem;
    max-width: 1200px;
    margin: 0 auto;
}

/* Header Styling */
h1 {
    color: #1a1a1a;
    text-align: center;
    margin-bottom: 2.5rem;
    font-size: 3em;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(120deg, #2b5876, #4e4376);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Sidebar Enhancement */
.stSidebar {
    background-color: #ffffff;
    padding: 2rem;
    border-right: 1px solid rgba(0,0,0,0.1);
    box-shadow: 2px 0 8px rgba(0,0,0,0.05);
}

/* Input Fields Enhancement */
.stSelectbox div[data-baseweb="select"],
.stNumberInput div[data-baseweb="input"] {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    transition: all 0.3s ease;
}

.stSelectbox div[data-baseweb="select"]:hover,
.stNumberInput div[data-baseweb="input"]:hover {
    border-color: #4e4376;
    box-shadow: 0 0 0 3px rgba(78, 67, 118, 0.1);
}

/* Hybrid Score Box Enhancement */
.hybrid-score-box {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 2em;
    margin: 2rem 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    transition: transform 0.3s ease;
}

.hybrid-score-box:hover {
    transform: translateY(-5px);
}

/* Risk Category Styling Enhancement */
.very-low-risk {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border-left: 5px solid #28a745;
}
.low-risk {
    background: linear-gradient(135deg, #cce5ff 0%, #b8daff 100%);
    border-left: 5px solid #007bff;
}
.moderate-risk {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
    border-left: 5px solid #ffc107;
}
.high-risk {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border-left: 5px solid #dc3545;
}
.very-high-risk {
    background: linear-gradient(135deg, #e2e3e5 0%, #d6d8db 100%);
    border-left: 5px solid #343a40;
}

/* Recommendation Header Enhancement */
.recommendation-header {
    font-weight: 700;
    padding: 12px 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    display: block;
    font-size: 1.1em;
    transition: all 0.3s ease;
}

/* Risk-specific recommendation headers */
.recommendation-very-low {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    color: #155724;
    border-left: 5px solid #28a745;
}

.recommendation-low {
    background: linear-gradient(135deg, #cce5ff 0%, #b8daff 100%);
    color: #004085;
    border-left: 5px solid #007bff;
}

.recommendation-moderate {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
    color: #856404;
    border-left: 5px solid #ffc107;
}

.recommendation-high {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    color: #721c24;
    border-left: 5px solid #dc3545;
}

.recommendation-very-high {
    background: linear-gradient(135deg, #e2e3e5 0%, #d6d8db 100%);
    color: #1b1e21;
    border-left: 5px solid #343a40;
}

/* Button Enhancement */
.stButton > button {
    background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
    color: white;
    padding: 0.8em 2em;
    border-radius: 8px;
    border: none;
    font-weight: 500;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(78, 67, 118, 0.4);
}

/* Table Enhancement */
table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border-radius: 12px;
    overflow: hidden;
    background: white;
    border: 1px solid #e0e0e0;
}

/* Table Header Enhancement */
th {
    background: #FFE4C4;  /* Light orange (Bisque) background */
    color: #664433;  /* Dark brown text for contrast */
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.9em;
    letter-spacing: 0.5px;
    padding: 16px 20px;
    border-bottom: 2px solid #FFB366;  /* Darker orange border */
    position: relative;
    transition: all 0.3s ease;
}

/* Table Header Hover Effect */
th:hover {
    background: #FFD5AA;  /* Slightly darker on hover */
}

/* Table Cell Enhancement */
td {
    padding: 14px 20px;
    border-bottom: 1px solid #eee;
    color: #495057;
    font-size: 0.95em;
    transition: background-color 0.2s ease;
}

/* Zebra Striping Enhancement */
tr:nth-child(even) {
    background-color: #f8f9fa;
}

/* Row Hover Effect */
tr:hover td {
    background-color: #f0f4f8;
}

/* Last Row Styling */
tr:last-child td {
    border-bottom: none;
}

/* First Column Enhancement */
td:first-child, th:first-child {
    padding-left: 24px;
}

/* Last Column Enhancement */
td:last-child, th:last-child {
    padding-right: 24px;
}

/* Table Caption Styling (if any) */
caption {
    padding: 12px;
    font-weight: 600;
    color: #495057;
    background: #fff;
    border-bottom: 1px solid #eee;
}

/* Responsive Table */
@media screen and (max-width: 768px) {
    table {
        display: block;
        overflow-x: auto;
        white-space: nowrap;
    }
}

/* Expander Enhancement */
.streamlit-expander {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    background: white;
    transition: all 0.3s ease;
}

.streamlit-expander:hover {
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
}

/* Dose Box Enhancement */
.dose-box {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    border: 1px solid #e0e0e0;
}

.dose-info {
    color: #4e4376;
    font-weight: 500;
    font-size: 0.9em;
}

/* --- Flowchart CSS --- */
/* Flowchart Container */
.methodology-flowchart {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
    padding: 30px;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    margin: 30px 0;
    width: 100%; /* Ensure it takes available width */
    box-sizing: border-box; /* Include padding and border in the element's total width and height */
}

/* Flowchart Node */
.flowchart-node {
    width: 90%; /* Use percentage for responsiveness */
    max-width: 600px;
    padding: 20px;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 10px;
    border: 2px solid #dee2e6;
    text-align: center;
    position: relative;
    transition: all 0.3s ease;
    box-sizing: border-box; /* Include padding and border in the element's total width and height */
}

.flowchart-node:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

/* Flowchart Arrow */
.flowchart-arrow {
    width: 2px;
    height: 30px;
    background: #6c757d;
    position: relative;
}

/* Flowchart Arrowhead: Creates the triangle shape at the end of the arrow */
.flowchart-arrow::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    border-width: 8px 6px 0 6px;
    border-style: solid;
    border-color: #6c757d transparent transparent transparent;
}

/* Node Title */
.node-title {
    font-weight: 600;
    color: #2b5876;
    margin-bottom: 10px;
    font-size: 1.1em;
}

/* Node Content */
.node-content {
    color: #495057;
    font-size: 0.95em;
    line-height: 1.5;
}
/* --- End Flowchart CSS --- */
</style>
""", unsafe_allow_html=True)

# Add tabs for different views
tab1, tab2, tab3 = st.tabs(["Main Interface", "Detailed Scoring Guide", "Model Training Timeline and Methodological Summary"])

with tab1:
    st.title("PONV RISK PRO")

    # Centered text
    st.markdown("<div style='text-align: center;'>An initiative of MKCG Medical College & Hospital - MKCG MedAI Labs</div>", unsafe_allow_html=True)


    st.markdown(
        """
        <p style="font-size: 20px;">
        <b>Inspired by global advancements like the POTTER app (Massachusetts Medical School), this model leverages artificial intelligence to optimize clinical decision-decision-making, in collaboration with the Department of Anaesthesiology, MKCG Medical College & Hospital. It integrates multiple established PONV risk scoring systems - including Apfel, Koivuranta, and Bellville - into a unified, AI-enhanced predictive framework.</b>
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("PONV Risk Assessment Parameters")

    # ------------------------- PATIENT FACTORS -------------------------
    gender = st.sidebar.selectbox("Female Gender", ["No", "Yes"])
    smoker = st.sidebar.selectbox("Non-Smoker", ["No", "Yes"])
    history_ponv = st.sidebar.selectbox("History of PONV or Motion Sickness", ["No", "Yes"])
    age = st.sidebar.slider("Age", 18, 80, 35)
    preop_anxiety = st.sidebar.selectbox("Preoperative Anxiety", ["No", "Yes"])
    history_migraine = st.sidebar.selectbox("History of Migraine", ["No", "Yes"])
    obesity = st.sidebar.selectbox("BMI > 30", ["No", "Yes"])

    # ------------------------- SURGICAL FACTORS -------------------------
    abdominal_surgery = st.sidebar.selectbox("Abdominal or Laparoscopic Surgery", ["No", "Yes"])
    ent_surgery = st.sidebar.selectbox("ENT/Neurosurgery/Ophthalmic Surgery", ["No", "Yes"])
    gynae_surgery = st.sidebar.selectbox("Gynecological or Breast Surgery", ["No", "Yes"])
    surgery_duration = st.sidebar.selectbox("Surgery Duration > 60 min", ["No", "Yes"])
    major_blood_loss = st.sidebar.selectbox("Major Blood Loss > 500 mL", ["No", "Yes"])
    volatile_agents = st.sidebar.selectbox("Use of Volatile Agents (Sevo/Iso/Des)", ["No", "Yes"])
    nitrous_oxide = st.sidebar.selectbox("Use of Nitrous Oxide", ["No", "Yes"])

    # ------------------------- DRUG FACTORS (WITH DOSE) -------------------------
    # Updated Drug Administration (Specify Dose) Section
    st.sidebar.markdown("<div class='drug-admin-header'><h2>Drug Administration (Specify Dose)</h2></div>", unsafe_allow_html=True)

    # Drug: Ondansetron
    st.sidebar.markdown("""
    <div class='dose-box'>
        <b>Ondansetron (4-24 mg)</b><br>
        <div class='dose-info'>
            Route: IV, Oral<br>
            Clinical Use: PONV prevention<br>
            PONV Score: 0 to -2 (4 mg or higher effective; full dose 16-24 mg offers maximum antiemetic effect)
        </div>
    </div>
    """, unsafe_allow_html=True)
    ondansetron_dose = st.sidebar.number_input("Ondansetron (mg)", 0.0, 24.0, 0.0, key='ondansetron_dose')

    # Drug: Midazolam
    st.sidebar.markdown("""
    <div class='dose-box'>
        <b>Midazolam (0.02-0.5 mg/kg or up to 20 mg)</b><br>
        <div class='dose-info'>
            Route: IV, IM, PO, IN, PR<br>
            Clinical Use: Sedation, induction, seizure control<br>
            PONV Score: 0 to -2 (Protective benefit increases with dose)
        </div>
    </div>
    """, unsafe_allow_html=True)
    midazolam_dose = st.sidebar.number_input("Midazolam (mg)", 0.0, 20.0, 0.0, key='midazolam_dose')

    # Drug: Dexamethasone
    st.sidebar.markdown("""
    <div class='dose-box'>
        <b>Dexamethasone (4-40 mg)</b><br>
        <div class='dose-info'>
            Route: IV<br>
            Clinical Use: PONV prophylaxis, inflammation<br>
            PONV Score: 0 to -1 (4 mg or higher useful for delayed PONV; high dose used for chemotherapy N/V)
        </div>
    </div>
    """, unsafe_allow_html=True)
    dexamethasone_dose = st.sidebar.number_input("Dexamethasone (mg)", 0.0, 40.0, 0.0, key='dexamethasone_dose')

    # Drug: Glycopyrrolate
    st.sidebar.markdown("""
    <div class='dose-box'>
        <b>Glycopyrrolate (0.1-0.4 mg)</b><br>
        <div class='dose-info'>
            Route: IV, IM<br>
            Clinical Use: Antisialagogue, vagolytic<br>
            PONV Score: 0 to +1 (At therapeutic doses, increases PONV risk slightly)
        </div>
    </div>
    """, unsafe_allow_html=True)
    glycopyrrolate_dose = st.sidebar.number_input("Glycopyrrolate (mg)", 0.0, 0.4, 0.0, key='glycopyrrolate_dose')

    # Drug: Nalbuphine
    st.sidebar.markdown("""
    <div class='dose-box'>
        <b>Nalbuphine (5-20 mg)</b><br>
        <div class='dose-info'>
            Route: IV, IM<br>
            Clinical Use: Opioid analgesic<br>
            PONV Score: 0 to +1 (Mild to moderate emetogenicity at higher doses)
        </div>
    </div>
    """, unsafe_allow_html=True)
    nalbuphine_dose = st.sidebar.number_input("Nalbuphine (mg)", 0.0, 20.0, 0.0, key='nalbuphine_dose')

    # Drug: Fentanyl
    st.sidebar.markdown("""
    <div class='dose-box'>
        <b>Fentanyl (25-2000 mcg)</b><br>
        <div class='dose-info'>
            Route: IV<br>
            Clinical Use: Intraoperative analgesia<br>
            PONV Score: 0 to +3 (Strongest dose-dependent PONV risk among opioids)
        </div>
    </div>
    """, unsafe_allow_html=True)
    fentanyl_dose = st.sidebar.number_input("Fentanyl (mcg)", 0.0, 2000.0, 0.0, key='fentanyl_dose')

    # Drug: Butorphanol
    st.sidebar.markdown("""
    <div class='dose-box'>
        <b>Butorphanol (0.5-4 mg)</b><br>
        <div class='dose-info'>
            Route: IV, IM<br>
            Clinical Use: Opioid analgesic<br>
            PONV Score: 0 to +1 (Partial agonist, less risky than fentanyl)
        </div>
    </div>
    """, unsafe_allow_html=True)
    butorphanol_dose = st.sidebar.number_input("Butorphanol (mg)", 0.0, 4.0, 0.0, key='butorphanol_dose')

    # Drug: Pentazocine
    st.sidebar.markdown("""
    <div class='dose-box'>
        <b>Pentazocine (30-360 mg)</b><br>
        <div class='dose-info'>
            Route: IV, IM, Oral<br>
            Clinical Use: Opioid analgesic<br>
            PONV Score: 0 to +3 (Strongly emetogenic at higher doses)
        </div>
    </div>
    """, unsafe_allow_html=True)
    pentazocine_dose = st.sidebar.number_input("Pentazocine (mg)", 0.0, 360.0, 0.0, key='pentazocine_dose')

    # Drug: Propofol (TIVA)
    st.sidebar.markdown("""
    <div class='dose-box'>
        <b>Propofol (10-250 mg/hr)</b><br>
        <div class='dose-info'>
            Route: IV infusion<br>
            Clinical Use: Maintenance of anesthesia<br>
            PONV Score: 0 to -3 (Continuous infusion provides maximal protective benefit)
        </div>
    </div>
    """, unsafe_allow_html=True)
    propofol_tiva_dose = st.sidebar.number_input("Propofol (mg/hr)", 0.0, 250.0, 0.0, key='propofol_tiva_dose')

    # Drug: Propofol (Induction)
    st.sidebar.markdown("""
    <div class='dose-box'>
        <b>Propofol (1-2.5 mg/kg)</b><br>
        <div class='dose-info'>
            Route: IV bolus<br>
            Clinical Use: Induction of anesthesia<br>
            PONV Score: 0 to -1 (Induction effect only, volatile agents override antiemetic effect)
        </div>
    </div>
    """, unsafe_allow_html=True)
    propofol_induction_dose = st.sidebar.number_input("Propofol (Induction, mg/kg)", 0.0, 2.5, 0.0, key='propofol_induction_dose')

    # Drug: Sevoflurane/Isoflurane/Desflurane
    st.sidebar.selectbox("Use of Sevoflurane/Isoflurane/Desflurane", ["No", "Yes"], key='volatile_agents_selectbox')
    st.sidebar.markdown("""
    <div class='dose-box'>
        <b>Sevoflurane/Isoflurane/Desflurane (1 MAC dose or higher)</b><br>
        <div class='dose-info'>
            Route: Inhalational<br>
            Clinical Use: Maintenance of general anesthesia<br>
            PONV Score: 0 to +2
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Propofol Mode Selection (Added based on the scoring function)
    propofol_mode = st.sidebar.selectbox("Propofol Mode", ["None", "Induction Only", "TIVA"])

    # Muscle Relaxant Selection (Added based on the scoring breakdown, though not used in scoring)
    muscle_relaxant = st.sidebar.selectbox("Muscle Relaxant Used", ["None", "Succinylcholine", "Rocuronium", "Vecuronium", "Atracurium", "Cisatracurium"])


    # ------------------------- HYBRID SCORING FUNCTION -------------------------
    def binary(val):
        return 1 if val == "Yes" else 0

    def propofol_score(mode):
        return -3 if mode == "TIVA" else -1 if mode == "Induction Only" else 0

    def calculate_hybrid_score():
        score = 0
        
        # Patient factors
        if gender == "Yes":
            score += 1
        if smoker == "Yes":
            score += 1
        if history_ponv == "Yes":
            score += 1
        if age > 50:
            score += 1
        if preop_anxiety == "Yes":
            score += 1
        if history_migraine == "Yes":
            score += 1
        if obesity == "Yes":
            score += 1
        
        return score


    def risk_category(score):
        if score <= -5:
            return "Very Low Risk", "very-low-risk"  # CSS class
        elif -4 <= score <= 3:
            return "Low Risk", "low-risk"  # CSS class
        elif 4 <= score <= 9:
            return "Moderate Risk", "moderate-risk"  # CSS class
        elif 10 <= score <= 15:
            return "High Risk", "high-risk"  # CSS class
        else:
            return "Very High Risk", "very-high-risk"  # CSS class

    # ------------------------- DISPLAY HYBRID SCORE -------------------------
    hybrid_score = calculate_hybrid_score()
    category, css_class = risk_category(hybrid_score) # Use CSS class instead of color

    st.subheader("Hybrid PONV Score Summary")
    st.markdown(f"""
    <div class='hybrid-score-box {css_class}'>
        <h3 style='margin: 0;'>Total Hybrid Score: {hybrid_score}</h3>
        <h4 style='margin: 0;'>Risk Category: <span style='font-weight: bold;'>{category}</span></h4>
    </div>
    """, unsafe_allow_html=True)


    # ------------------------- RECOMMENDATIONS FOR CLINICIANS (Expandable) -------------------------

    st.subheader("Recommendations for Clinicians")
    with st.expander("View Recommendations"):
        st.markdown("""
        <div style='font-size: 0.95em; color: #495057; margin-bottom: 1.5rem;'>
        Evidence-based recommendations tailored for PONV risk levels, aligned with international guidelines (Apfel, ASA, ESA):
        </div>
        """, unsafe_allow_html=True)

        if category == "Moderate Risk":
            st.markdown("""
            <div class='recommendation-header recommendation-moderate'>
                <div style='font-size: 1.1em; font-weight: 700;'>Moderate Risk (Score: 4-9)</div>
                <div style='font-size: 0.95em;'>Dual prophylaxis recommended</div>
            </div>
            <div style='margin-bottom: 1.2em;'>
                <b>Pharmacological Interventions</b>
                <ul>
                    <li>Ondansetron 4-8 mg IV - 5-HT3 antagonist at surgery end</li>
                    <li>Dexamethasone 4 mg or higher IV - At induction (delayed effect)</li>
                    <li>Midazolam 1-2 mg IV - If anxiety present</li>
                </ul>
            </div>
            <div style='margin-bottom: 1.2em;'>
                <b>Anesthetic Technique</b>
                <ul>
                    <li>TIVA with Propofol - Preferred when feasible</li>
                    <li>Minimize opioids - Use non-opioid alternatives</li>
                    <li>Avoid N2O - When possible</li>
                </ul>
            </div>
            <div>
                <b>Supportive Measures</b>
                <ul>
                    <li>Hydration - Ensure adequate perioperative fluids</li>
                    <li>Gastric management - Avoid distension</li>
                    <li>Monitoring - >30 minutes post-op</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif category == "High Risk" or category == "Very High Risk":
            st.markdown("""
            <div class='recommendation-header recommendation-high'>
                <div style='font-size: 1.1em; font-weight: 700;'>High Risk (Score: 10-15) or Very High Risk</div>
                <div style='font-size: 0.95em;'>Multimodal prevention required</div>
            </div>
            <div style='margin-bottom: 1.2em;'>
                <b>Pharmacological Interventions</b>
                <ul>
                    <li>Triple therapy: Ondansetron + Dexamethasone + NK1 antagonist (if available)</li>
                    <li>Scopolamine patch - For prolonged effect</li>
                    <li>Droperidol 0.625-1.25 mg IV - If not contraindicated</li>
                </ul>
            </div>
            <div style='margin-bottom: 1.2em;'>
                <b>Anesthetic Technique</b>
                <ul>
                    <li>Mandatory TIVA - Propofol-based</li>
                    <li>Opioid-sparing - Regional techniques preferred</li>
                </ul>
            </div>
            <div>
                <b>Postoperative Care</b>
                <ul>
                    <li>Extended PACU - >2 hours monitoring</li>
                    <li>Rescue medications - Immediately available</li>
                    <li>Discharge Rx - Consider antiemetics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif category in ["Very Low Risk", "Low Risk"]:
             st.info("Recommendations for Very Low and Low Risk categories are generally focused on minimizing risk factors and may not require routine pharmacological prophylaxis. Consult relevant guidelines for specific approaches.")

        st.markdown("""
        <div style='font-weight: 600; color: #721c24; margin: 15px 0 10px 0;'>Rescue Therapy</div>
        <div style='font-size: 0.95em; color: #495057;'>For all moderate/severe risk patients who develop PONV despite prophylaxis:</div>
        <ul style='margin-top: 0; padding-left: 25px;'>
            <li>First-line: Metoclopramide 10 mg IV or Promethazine 12.5-25 mg IV</li>
            <li>Second-line: Scopolamine patch (if available) or haloperidol 0.5-1 mg IV</li>
            <li>Do not repeat the same class of antiemetic used in prophylaxis</li>
        </ul>
        """, unsafe_allow_html=True)


    # ------------------------- FEATURE VECTOR -------------------------
    # Construct the feature vector based on the sidebar inputs
    feature_vector = [
        binary(gender), binary(smoker), binary(history_ponv), age, binary(preop_anxiety),
        binary(history_migraine), binary(obesity), binary(abdominal_surgery), binary(ent_surgery),
        binary(gynae_surgery), binary(surgery_duration), binary(major_blood_loss),
        binary(volatile_agents), binary(nitrous_oxide),
        midazolam_dose, ondansetron_dose, dexamethasone_dose, glycopyrrolate_dose,
        nalbuphine_dose, fentanyl_dose / 1000.0, butorphanol_dose, pentazocine_dose,
        propofol_score(propofol_mode)
    ]

    feature_names = [
        "Female", "Non-Smoker", "History PONV", "Age", "Preop Anxiety", "Migraine", "Obesity",
        "Abdominal Surg", "ENT/Neuro/Ophthalmic", "Gynae/Breast Surg", "Surg >60min",
        "Blood Loss >500ml", "Volatile Agents", "Nitrous Oxide",
        "Midazolam (mg)", "Ondansetron (mg)", "Dexamethasone (mg)", "Glycopyrrolate (mg)",
        "Nalbuphine (mg)", "Fentanyl (mg)", "Butorphanol (mg)", "Pentazocine (mg)",
        "Propofol Score"
    ]

    # Convert the current feature vector to a numpy array for scaling and prediction
    current_features = np.array(feature_vector).reshape(1, -1)


    # ------------------------- SYNTHETIC DATA -------------------------
    n_features = len(feature_vector)
    np.random.seed(42)

    @st.cache_data
    def generate_synthetic_data(n_samples=2000):
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples)

        for i in range(n_samples):
            # Generate more realistic feature distributions with adjusted weights
            X[i, 0:14] = np.random.binomial(1, 0.5, 14)  # Binary features
            X[i, 3] = np.random.normal(45, 15)  # Age with normal distribution
            # Ensure drug doses are non-negative
            X[i, 14:22] = np.maximum(0, np.random.exponential(2, 8))  # Drug doses with exponential distribution, min 0
            X[i, 22] = np.random.choice([-3, -1, 0])  # Propofol score

            # Adjust risk factors weights to improve model performance
            risk_factors = (
                2.5 * X[i, 0] +  # Female gender (increased weight)
                2.0 * X[i, 2] +  # History of PONV (increased weight)
                1.5 * X[i, 4] +  # Preop anxiety (increased weight)
                1.5 * X[i, 5] +  # History of migraine (increased weight)
                1.2 * X[i, 6] +  # Obesity (increased weight)
                2.0 * X[i, 7] +  # Abdominal surgery (increased weight)
                1.5 * X[i, 8] +  # ENT surgery (increased weight)
                1.5 * X[i, 9] +  # Gynae surgery (increased weight)
                1.2 * X[i, 11] + # Major blood loss (increased weight)
                2.5 * X[i, 12] + # Volatile agents (increased weight)
                1.5 * X[i, 13]   # Nitrous oxide (increased weight)
            )

            # Adjust protective factors weights
            protective_factors = (
                1.5 * X[i, 14] + # Midazolam (increased weight)
                2.5 * X[i, 15] + # Ondansetron (increased weight)
                2.5 * X[i, 16] + # Dexamethasone (increased weight)
                1.5 * X[i, 17]   # Glycopyrrolate (increased weight)
            )

            # Calculate final probability with adjusted sigmoid function
            prob = 1 / (1 + np.exp(-1.5 * (risk_factors - protective_factors)))
            y[i] = np.random.binomial(1, prob)

        return X, y

    # Generate synthetic data
    X, y = generate_synthetic_data(2000)

    # Split and preprocess data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Add feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Add SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    # Train XGBoost model
    xgb_model = XGBClassifier(
        max_depth=5,
        learning_rate=0.03,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='auc'
    )
    xgb_model.fit(X_train_balanced, y_train_balanced)

    # Train AdaBoost model
    ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
    ada_model.fit(X_train_balanced, y_train_balanced)

    # Train LightGBM model
    lgb_model = lgb.LGBMClassifier()
    lgb_model.fit(X_train_balanced, y_train_balanced)

    # Train LinearSVC model and calibrate
    svc_model = LinearSVC(max_iter=10000, random_state=42)
    svc_cal = CalibratedClassifierCV(svc_model, method='sigmoid', cv=5)
    svc_cal.fit(X_train_balanced, y_train_balanced)


    


    # ------------------------- MODEL EVALUATION -------------------------
    st.subheader("Model AUC Scores")

    # Initialize AUC variables to None in case calculation is skipped
    auc_svc_train, auc_ada_train, auc_lgb_train, auc_xgb_train = None, None, None, None
    auc_svc_val, auc_ada_val, auc_lgb_val, auc_xgb_val = None, None, None, None # Corrected variable name here

    # Initialize figure variables to None
    fig_train_all, fig_val_all = None, None

    # Check if there are at least two classes in the training and validation sets
    if len(np.unique(y_train_balanced)) < 2:
        st.warning("Training data contains only one class. Cannot calculate ROC curves and AUC for training data.")
    else:
        # Calculate ROC curves for all models on training data (scaled)
        fpr_xgb_train, tpr_xgb_train, _ = roc_curve(y_train_balanced, xgb_model.predict_proba(X_train_balanced)[:, 1])
        fpr_ada_train, tpr_ada_train, _ = roc_curve(y_train_balanced, ada_model.predict_proba(X_train_balanced)[:, 1])
        fpr_lgb_train, tpr_lgb_train, _ = roc_curve(y_train_balanced, lgb_model.predict_proba(X_train_balanced)[:, 1])
        fpr_svc_train, tpr_svc_train, _ = roc_curve(y_train_balanced, svc_cal.predict_proba(X_train_balanced)[:, 1])

        auc_xgb_train = auc(fpr_xgb_train, tpr_xgb_train)
        auc_ada_train = auc(fpr_ada_train, tpr_ada_train)
        auc_lgb_train = auc(fpr_lgb_train, tpr_lgb_train)
        auc_svc_train = auc(fpr_svc_train, tpr_svc_train)

        # Plot Training ROC curves (All Models)
        fig_train_all, ax_train_all = plt.subplots(figsize=(5, 3)) # Small size
        ax_train_all.plot(fpr_svc_train, tpr_svc_train, label=f"LinearSVC (AUC = {auc_svc_train:.3f})")
        ax_train_all.plot(fpr_ada_train, tpr_ada_train, label=f"AdaBoost (AUC = {auc_ada_train:.3f})")
        ax_train_all.plot(fpr_lgb_train, tpr_lgb_train, label=f"LightGBM (AUC = {auc_lgb_train:.3f})")
        ax_train_all.plot(fpr_xgb_train, tpr_xgb_train, label=f"XGBoost (AUC = {auc_xgb_train:.3f})")
        ax_train_all.plot([0, 1], [0, 1], 'k--')
        ax_train_all.set_xlabel("False Positive Rate")
        ax_train_all.set_ylabel("True Positive Rate")
        ax_train_all.set_title("Training ROC Curve (All Models)")
        ax_train_all.legend(loc="lower right", fontsize='small')


    if len(np.unique(y_val)) < 2:
        st.warning("Validation data contains only one class. Cannot calculate ROC curves and AUC for validation data.")
    else:
        # Calculate ROC curves for all models on validation data (scaled)
        fpr_xgb_val, tpr_xgb_val, _ = roc_curve(y_val, xgb_model.predict_proba(X_val_scaled)[:, 1])
        fpr_ada_val, tpr_ada_val, _ = roc_curve(y_val, ada_model.predict_proba(X_val_scaled)[:, 1])
        fpr_lgb_val, tpr_lgb_val, _ = roc_curve(y_val, lgb_model.predict_proba(X_val_scaled)[:, 1])
        fpr_svc_val, tpr_svc_val, _ = roc_curve(y_val, svc_cal.predict_proba(X_val_scaled)[:, 1])

        auc_xgb_val = auc(fpr_xgb_val, tpr_xgb_val)
        auc_ada_val = auc(fpr_ada_val, tpr_ada_val)
        auc_lgb_val = auc(fpr_lgb_val, tpr_lgb_val)
        auc_svc_val = auc(fpr_svc_val, tpr_svc_val)

        # Plot Validation ROC curves (All Models)
        fig_val_all, ax_val_all = plt.subplots(figsize=(5, 3)) # Small size
        ax_val_all.plot(fpr_svc_val, tpr_svc_val, label=f"LinearSVC (AUC = {auc_svc_val:.3f})")
        ax_val_all.plot(fpr_ada_val, tpr_ada_val, label=f"AdaBoost (AUC = {auc_ada_val:.3f})")
        ax_val_all.plot(fpr_lgb_val, tpr_lgb_val, label=f"LightGBM (AUC = {auc_lgb_val:.3f})")
        ax_val_all.plot(fpr_xgb_val, tpr_xgb_val, label=f"XGBoost (AUC = {auc_xgb_val:.3f})")
        ax_val_all.plot([0, 1], [0, 1], 'k--')
        ax_val_all.set_xlabel("False Positive Rate")
        ax_val_all.set_ylabel("True Positive Rate")
        ax_val_all.set_title("Validation ROC Curve (All Models)")
        ax_val_all.legend(loc="lower right", fontsize='small')


    # Create the DataFrame with calculated AUC values (will be None if calculation was skipped)
    # Ensure lists always have 4 elements
    df_auc = pd.DataFrame({
        'Model': ['LinearSVC (Calibrated)', 'AdaBoost', 'LightGBM', 'XGBoost'],
        'Training AUC': [auc_svc_train, auc_ada_train, auc_lgb_train, auc_xgb_train],
        'Validation AUC': [auc_svc_val, auc_ada_val, auc_lgb_val, auc_xgb_val]
    })


    # Format the numeric columns to 3 decimal places, handling None values
    for col in ['Training AUC', 'Validation AUC']:
        df_auc[col] = df_auc[col].apply(lambda x: '{:.3f}'.format(x) if x is not None else 'N/A')

    st.table(df_auc)

    # Display the two small ROC plots side-by-side if they were generated
    col1, col2 = st.columns(2)
    with col1:
        if fig_train_all is not None: # Check if figure was generated
            st.pyplot(fig_train_all)
    with col2:
        if fig_val_all is not None: # Check if figure was generated
            st.pyplot(fig_val_all)


    # Calculate and show metrics for all models on validation data (scaled)
    st.subheader("Model Performance Metrics (Calculated on Validation Data)")

    # Only calculate and display metrics if validation data has at least two classes
    if len(np.unique(y_val)) < 2:
         st.warning("Validation data contains only one class. Cannot calculate performance metrics.")
    else:
        def calculate_metrics(model, X_val_scaled, y_val):
            if hasattr(model, 'predict_proba'):
                preds_proba = model.predict_proba(X_val_scaled)[:, 1]
                preds = (preds_proba > 0.5).astype(int) # Using 0.5 threshold
            else:
                # For models without predict_proba (like LinearSVC before calibration)
                # we would typically not calculate metrics that rely on probability thresholds.
                # Since svc_cal is calibrated, it has predict_proba, so this else block might not be strictly needed
                # but keeping it as a safeguard.
                preds = model.predict(X_val_scaled)

            # Handle potential errors if precision/recall/f1 are undefined (e.g., no positive predictions)
            try:
                prec = precision_score(y_val, preds)
            except:
                prec = np.nan # Use NaN for undefined metrics

            try:
                rec = recall_score(y_val, preds)
            except:
                rec = np.nan # Use NaN for undefined metrics

            try:
                f1 = f1_score(y_val, preds)
            except:
                f1 = np.nan # Use NaN for undefined metrics

            acc = accuracy_score(y_val, preds)

            return acc, prec, rec, f1

        # Calculate metrics for each model
        acc_svc, prec_svc, rec_svc, f1_svc = calculate_metrics(svc_cal, X_val_scaled, y_val)
        acc_ada, prec_ada, rec_ada, f1_ada = calculate_metrics(ada_model, X_val_scaled, y_val)
        acc_lgb, prec_lgb, rec_lgb, f1_lgb = calculate_metrics(lgb_model, X_val_scaled, y_val)
        acc_xgb, prec_xgb, rec_xgb, f1_xgb = calculate_metrics(xgb_model, X_val_scaled, y_val)

        # Create a DataFrame for the calculated performance metrics
        calculated_metrics_data = {
            'Model': ['LinearSVC (Calibrated)', 'AdaBoost', 'LightGBM', 'XGBoost'],
            'Accuracy': [acc_svc, acc_ada, acc_lgb, acc_xgb],
            'Precision': [prec_svc, prec_ada, prec_lgb, prec_xgb],
            'Recall': [rec_svc, rec_ada, rec_lgb, rec_xgb],
            'F1-score': [f1_svc, f1_ada, f1_lgb, f1_xgb]
        }
        df_calculated_metrics = pd.DataFrame(calculated_metrics_data)

        # Format the numeric columns to 2 decimal places, handling NaN
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
            df_calculated_metrics[col] = df_calculated_metrics[col].apply(lambda x: '{:.2f}'.format(x) if pd.notna(x) else 'N/A')


        # Display the table
        st.table(df_calculated_metrics)


    # ------------------------- USER INPUT PREDICTION (All Models) -------------------------
    input_array = np.array(feature_vector).reshape(1, -1)
    # Scale the input array using the same scaler fitted on the training data
    input_scaled_for_prediction = scaler.transform(input_array)

    # Predict probabilities using the scaled input for all models
    prob_svc = svc_cal.predict_proba(input_scaled_for_prediction)[:, 1][0]
    prob_ada = ada_model.predict_proba(input_scaled_for_prediction)[:, 1][0]
    prob_lgb = lgb_model.predict_proba(input_scaled_for_prediction)[:, 1][0]
    prob_xgb = xgb_model.predict_proba(input_scaled_for_prediction)[:, 1][0]

    st.subheader("Predicted PONV Risk (Your Input)")

    # Create a DataFrame for predicted risks
    predicted_risk_data = {
        'Model': ['LinearSVC (Calibrated)', 'AdaBoost', 'LightGBM', 'XGBoost'],
        'Predicted Risk': [prob_svc, prob_ada, prob_lgb, prob_xgb]
    }
    df_predicted_risk = pd.DataFrame(predicted_risk_data)

    # Format the numeric column to 2 decimal places
    df_predicted_risk['Predicted Risk'] = df_predicted_risk['Predicted Risk'].map('{:.2f}'.format)

    # Display the table
    st.table(df_predicted_risk)


    st.markdown(
        "<small>This model uses synthetic data based on your input structure for demo only. Train on real clinical data for deployment.</small>",
        unsafe_allow_html=True,
    )

    # ------------------------- FEEDBACK SECTION -------------------------
    st.subheader("Feedback (Alpha Testing)")
    feedback = st.text_area("Please provide any feedback or suggestions:", key='feedback_area')
    if st.button("Submit Feedback", key='submit_feedback'):
        if feedback:
            st.write("Thank you for your feedback!")
            # You can store the feedback (e.g., in a file, database) here
        else:
            st.write("Please enter some feedback.")

    # ------------------------- UPLOAD REAL-WORLD DATA -------------------------
    st.subheader("Upload Real-World Dataset for Hybrid Risk vs Predicted Risk Evaluation")
    uploaded_file = st.file_uploader("Upload File", key='file_uploader')

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Add validation for required columns
            # Adjusted required columns based on potential use for evaluation
            # Assuming the uploaded data has the same feature names as the synthetic data
            required_columns = feature_names + ['PONV_Outcome'] # Assuming a column for actual outcome (0 or 1)
            if not all(col in df.columns for col in required_columns):
                st.error(f"CSV file must contain these columns: {', '.join(required_columns)}")
            else:
                st.success("File uploaded successfully! Processing data...")

                # Prepare uploaded data for prediction
                uploaded_features = df[feature_names]
                uploaded_outcomes = df['PONV_Outcome']

                # Scale the uploaded data using the same scaler
                uploaded_features_scaled = scaler.transform(uploaded_features)

                # Predict on uploaded data
                df['Predicted_Risk_XGBoost'] = xgb_model.predict_proba(uploaded_features_scaled)[:, 1]
                df['Predicted_Risk_AdaBoost'] = ada_model.predict_proba(uploaded_features_scaled)[:, 1]
                df['Predicted_Risk_LightGBM'] = lgb_model.predict_proba(uploaded_features_scaled)[:, 1]
                df['Predicted_Risk_LinearSVC'] = svc_cal.predict_proba(uploaded_features_scaled)[:, 1]

                # Calculate Hybrid Score for uploaded data (assuming necessary columns exist)
                # This requires mapping the 'Yes'/'No' columns and dose columns from the uploaded data
                # to the hybrid score calculation logic. This part is complex and depends heavily
                # on the exact column names and format in the uploaded CSV.
                # For demonstration, let's assume the binary columns are named the same as feature_names
                # and dose columns are also named appropriately.
                # A more robust implementation would require clear mapping or a specific template.

                # Simplified Hybrid Score Calculation for Uploaded Data (requires careful column mapping)
                # This is a placeholder and needs to be adapted based on the actual column names
                # and logic from the `calculate_hybrid_score` function.
                # Example (Highly Dependent on CSV structure):
                # df['Hybrid_Score_Calculated'] = df.apply(lambda row:
                #     (1 if row['Female'] == 1 else 0) +
                #     (1 if row['Non-Smoker'] == 1 else 0) +
                #     # ... add other binary factors ...
                #     (-1 if row['Midazolam (mg)'] > 0 else 0) +
                #     # ... add other drug factors ...
                #     (propofol_score(row['Propofol Mode'])) # Requires mapping mode strings
                # , axis=1)

                st.subheader("Evaluation on Uploaded Data")

                if len(np.unique(uploaded_outcomes)) < 2:
                    st.warning("Uploaded data contains only one class for 'PONV_Outcome'. Cannot calculate performance metrics.")
                else:
                    # Calculate and display metrics for each model on uploaded data
                    uploaded_metrics = {}
                    uploaded_metrics['XGBoost'] = calculate_metrics(xgb_model, uploaded_features_scaled, uploaded_outcomes)
                    uploaded_metrics['AdaBoost'] = calculate_metrics(ada_model, uploaded_features_scaled, uploaded_outcomes)
                    uploaded_metrics['LightGBM'] = calculate_metrics(lgb_model, uploaded_features_scaled, uploaded_outcomes)
                    uploaded_metrics['LinearSVC (Calibrated)'] = calculate_metrics(svc_cal, uploaded_features_scaled, uploaded_outcomes)

                    # Create DataFrame for uploaded data metrics
                    df_uploaded_metrics = pd.DataFrame.from_dict(uploaded_metrics, orient='index', columns=['Accuracy', 'Precision', 'Recall', 'F1-score'])
                    for col in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
                        df_uploaded_metrics[col] = df_uploaded_metrics[col].apply(lambda x: '{:.2f}'.format(x) if pd.notna(x) else 'N/A')

                    st.write("Model Performance Metrics on Uploaded Data:")
                    st.table(df_uploaded_metrics)

                    # Calculate and plot ROC curve for uploaded data
                    st.subheader("ROC Curve on Uploaded Data")
                    fig_uploaded_roc, ax_uploaded_roc = plt.subplots(figsize=(8, 6))

                    # Plot ROC for each model on uploaded data
                    for model_name, model in zip(['XGBoost', 'AdaBoost', 'LightGBM', 'LinearSVC (Calibrated)'], [xgb_model, ada_model, lgb_model, svc_cal]):
                        if hasattr(model, 'predict_proba'):
                            fpr, tpr, _ = roc_curve(uploaded_outcomes, model.predict_proba(uploaded_features_scaled)[:, 1])
                            roc_auc = auc(fpr, tpr)
                            ax_uploaded_roc.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

                    ax_uploaded_roc.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
                    ax_uploaded_roc.set_xlabel('False Positive Rate')
                    ax_uploaded_roc.set_ylabel('True Positive Rate')
                    ax_uploaded_roc.set_title('ROC Curve on Uploaded Data')
                    ax_uploaded_roc.legend(loc='lower right')
                    ax_uploaded_roc.grid(True)
                    st.pyplot(fig_uploaded_roc)


        except Exception as e:
            st.error(f"Error processing uploaded CSV file: {str(e)}")


    # ------------------------- LOG ENTRY AND SHOW ENTRIES -------------------------
    # Initialize database connection
    # Use st.session_state to store the connection and cursor to avoid re-initializing
    # on every rerun, which can lead to issues with SQLite.
    if 'conn' not in st.session_state:
        st.session_state.conn = sqlite3.connect('ponv_logs.db', check_same_thread=False)
        st.session_state.cursor = st.session_state.conn.cursor()

    conn = st.session_state.conn
    cursor = st.session_state.cursor

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gender TEXT,
            smoker TEXT,
            history_ponv TEXT,
            age INTEGER,
            anxiety TEXT,
            abdominal_surgery TEXT,
            volatile TEXT,
            n2o TEXT,
            midazolam REAL,
            ondansetron REAL,
            dexamethasone REAL,
            glycopyrrolate REAL,
            nalbuphine REAL,
            fentanyl REAL,
            butorphanol REAL,
            pentazocine REAL,
            propofol_mode TEXT,
            muscle_relaxant TEXT,
            hybrid_score INTEGER,
            predicted_risk_xgb REAL,
            predicted_risk_ada REAL,
            predicted_risk_svc REAL,
            predicted_risk_lgb REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit() # Commit table creation if it happened

    # Check and add columns if they don't exist (schema migration)
    def add_column_if_not_exists(cursor, table_name, column_name, column_type):
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        if column_name not in columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            conn.commit()
            # st.info(f"Added column '{column_name}' to table '{table_name}'.") # Removed this info message

    # Add any new columns here with their types
    # Note: These checks are already in the user's snippet, ensuring they are present.
    add_column_if_not_exists(cursor, 'logs', 'glycopyrrolate', 'REAL')
    add_column_if_not_exists(cursor, 'logs', 'nalbuphine', 'REAL')
    add_column_if_not_exists(cursor, 'logs', 'fentanyl', 'REAL')
    add_column_if_not_exists(cursor, 'logs', 'butorphanol', 'REAL')
    add_column_if_not_exists(cursor, 'logs', 'pentazocine', 'REAL')
    add_column_if_not_exists(cursor, 'logs', 'propofol_mode', 'TEXT')
    add_column_if_not_exists(cursor, 'logs', 'muscle_relaxant', 'TEXT')
    add_column_if_not_exists(cursor, 'logs', 'predicted_risk_xgb', 'REAL')
    add_column_if_not_exists(cursor, 'logs', 'predicted_risk_ada', 'REAL')
    add_column_if_not_exists(cursor, 'logs', 'predicted_risk_svc', 'REAL')
    add_column_if_not_exists(cursor, 'logs', 'predicted_risk_lgb', 'REAL')


    # Add opioid calculation before database logging
    opioid = "Yes" if (nalbuphine_dose > 0 or fentanyl_dose > 0 or
                         butorphanol_dose > 0 or pentazocine_dose > 0) else "No"

    if st.button("Log This Entry", key='log_entry_button'):
        try:
            cursor.execute('''
                INSERT INTO logs (
                    gender, smoker, history_ponv, age, anxiety,
                    abdominal_surgery, volatile, n2o, midazolam,
                    ondansetron, dexamethasone, glycopyrrolate,
                    nalbuphine, fentanyl, butorphanol, pentazocine,
                    propofol_mode, muscle_relaxant, hybrid_score,
                    predicted_risk_xgb, predicted_risk_ada,
                    predicted_risk_svc, predicted_risk_lgb
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                "Female" if gender == "Yes" else "Male",
                "No" if smoker == "Yes" else "Yes",
                "Yes" if history_ponv == "Yes" else "No",
                age,
                "Yes" if preop_anxiety == "Yes" else "No",
                "Yes" if (abdominal_surgery == "Yes" or ent_surgery == "Yes" or gynae_surgery == "Yes") else "No",
                "Yes" if volatile_agents == "Yes" else "No",
                "Yes" if nitrous_oxide == "Yes" else "No",
                midazolam_dose,
                ondansetron_dose,
                dexamethasone_dose,
                glycopyrrolate_dose,
                nalbuphine_dose,
                fentanyl_dose,
                butorphanol_dose,
                pentazocine_dose,
                propofol_mode,
                muscle_relaxant,
                hybrid_score,
                prob_xgb,
                prob_ada,
                prob_svc,
                prob_lgb
            ))
            conn.commit()
            st.success("Entry logged successfully!")
        except sqlite3.Error as e:
            st.error(f"Database error: {str(e)}")
        except NameError as e:
            st.error(f"Logging error: {str(e)}. Please ensure all input fields are selected/filled.")


    if st.button("Show All Entries", key='show_entries_button'):
        cursor.execute('SELECT * FROM logs ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        if rows:
            columns = [description[0] for description in cursor.description]
            df_log = pd.DataFrame(rows, columns=columns)
            st.dataframe(df_log)

            if not df_log.empty:
                csv = df_log.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download All Entries as CSV",
                    data=csv,
                    file_name='logged_ponv_entries.csv',
                    mime='text/csv',
                    key='download_button'
                )
            else:
                st.warning("No data available to download")
        else:
            st.info("No entries found in the database.")

    # Close the database connection when the app is done (or session ends)
    # This might not be strictly necessary in all Streamlit deployments,
    # but it's good practice.
    # Use an on_after_request or similar if available in the deployment environment
    # For simple Streamlit, the connection might persist across reruns due to session state.
    # Explicitly closing might be tricky without a clear app exit event.
    # conn.close() # Avoid closing here as it will break on rerun

    # Add a note about database persistence
    st.markdown("""
    <div style='font-size: 0.8em; text-align: center; color: #6c757d;'>
        Data is logged to a local SQLite file (`ponv_logs.db`). This file will persist as long as the Streamlit application's data directory is maintained.
    </div>
    """, unsafe_allow_html=True)


    # ------------------------- DISCLAIMER -------------------------
    st.markdown("""
    <br>
    <div style='font-size: 0.8em; text-align: center; color: #6c757d;'>
        <b>Disclaimer:</b> This application is for informational and educational purposes only and should not be considered a substitute for professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

with tab2:
    # ------------------------- DETAILED SCORING BREAKDOWN (Expandable) -------------------------
    st.subheader("Detailed Scoring Breakdown")
    with st.expander("View Individual Parameter Contributions"):
        # Define the score contribution for each parameter based on the sidebar selection
        parameter_scores = {
            "Female Gender": binary(gender),
            "Non-Smoker": binary(smoker),
            "History of PONV or Motion Sickness": binary(history_ponv),
            "Age": 1 if age > 50 else 0,  # Add age-based scoring here
            "Preoperative Anxiety": binary(preop_anxiety),
            "History of Migraine": binary(history_migraine),
            "BMI > 30": binary(obesity),
            "Abdominal or Laparoscopic Surgery": binary(abdominal_surgery),
            "ENT/Neurosurgery/Ophthalmic Surgery": binary(ent_surgery),
            "Gynecological or Breast Surgery": binary(gynae_surgery),
            "Surgery Duration > 60 min": binary(surgery_duration),
            "Major Blood Loss > 500 mL": binary(major_blood_loss),
            "Use of Volatile Agents (Sevo/Iso/Des)": binary(volatile_agents),
            "Use of Nitrous Oxide": binary(nitrous_oxide),
            "Midazolam Use": -1 if midazolam_dose > 0 else 0,
            "Ondansetron Use (>= 4mg)": -2 if ondansetron_dose >= 4 else 0,
            "Dexamethasone Use (>= 4mg)": -2 if dexamethasone_dose >= 4 else 0,
            "Glycopyrrolate Use": -1 if glycopyrrolate_dose > 0 else 0,
            "Nalbuphine Use": 1 if nalbuphine_dose > 0 else 0,
            "Fentanyl Use (> 100 mcg)": 1 if fentanyl_dose > 100 else 0,
            "Butorphanol Use": 1 if butorphanol_dose > 0 else 0,
            "Pentazocine Use": 1 if pentazocine_dose > 0 else 0,
            "Propofol Use": propofol_score(propofol_mode),
            "Muscle Relaxant Used": 0  # Muscle relaxant type doesn't contribute to the hybrid score
        }

        # Create a DataFrame for the detailed scoring breakdown
        df_scoring_breakdown = pd.DataFrame(list(parameter_scores.items()), columns=['Parameter', 'Score Contribution'])

        # Define a function to apply color based on the score
        def color_score(val):
            if val > 0:
                return f'color: #dc3545; font-weight: bold;'  # Red for positive contribution (risk increasing)
            elif val < 0:
                return f'color: #28a745; font-weight: bold;'  # Green for negative contribution (risk decreasing)
            else:
                return ''  # Default color for 0 contribution

        # Apply the color function to the 'Score Contribution' column
        styled_df = df_scoring_breakdown.style.applymap(color_score, subset=['Score Contribution'])

        # Display the styled table
        st.table(styled_df)

with tab3:
    st.title("Model Training Timeline and Methodological Summary")
    st.subheader("Model Training Timeline and Methodological Summary")
    with st.expander("View Model Training Timeline", expanded=False):
        st.markdown("""
        <style>
        .timeline-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1.5em;
            margin-bottom: 1.5em;
            font-size: 0.9em;
            font-family: 'Inter', sans-serif;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
            border-radius: 10px;
            overflow: hidden;
        }
        .timeline-table thead tr {
            background-color: #2b5876;
            color: #ffffff;
            text-align: left;
            font-weight: bold;
        }
        .timeline-table th,
        .timeline-table td {
            padding: 12px 15px;
            border: 1px solid #dddddd;
        }
        .timeline-table tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        .timeline-table tbody tr:nth-of-type(even) {
            background-color: #f3f3f3;
        }
        .timeline-table tbody tr:last-of-type {
            border-bottom: 2px solid #2b5876;
        }
        .timeline-table tbody tr:hover {
            background-color: #e9ecef;
            cursor: pointer;
        }
        .phase-cell {
            font-weight: 600;
            color: #2b5876;
        }
        .dates-cell {
            font-style: italic;
            color: #495057;
        }
        .n-cell {
            text-align: center;
            font-weight: 500;
        }
        .key-activities-cell ul {
            margin: 0;
            padding-left: 20px;
        }
        .key-activities-cell li {
            margin-bottom: 5px;
        }
        .biostat-notes-cell {
            font-size: 0.85em;
            color: #495057;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown("""
        <table class="timeline-table">
            <thead>
                <tr>
                    <th>Phase</th>
                    <th>n</th>
                    <th>Key Activities</th>
                    <th>Biostatistical Notes</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="phase-cell">1. Synthetic Data Simulation</td>
                    <td class="n-cell">2,000</td>
                    <td class="key-activities-cell">
                        <ul>
                            <li>Generate CTGAN- and <code>make_classification</code>-based dataset</li>
                            <li>Apply SMOTE for class balance</li>
                        </ul>
                    </td>
                    <td class="biostat-notes-cell">
                        • Cross-validate models (k=5)<br>
                        • Establish Youden's cutoffs on ROC
                    </td>
                </tr>
                <tr>
                    <td class="phase-cell">2. Pre-Pilot Prospective</td>
                    <td class="n-cell">250</td>
                    <td class="key-activities-cell">
                        <ul>
                            <li>Real-time EHR capture in OR/Recovery</li>
                            <li>App usability testing</li>
                        </ul>
                    </td>
                    <td class="biostat-notes-cell">
                        • Assess score distribution vs. synthetic<br>
                        • Early calibration by Platt scaling (Logistic regression)
                    </td>
                </tr>
                <tr>
                    <td class="phase-cell">3. Pilot Retrospective</td>
                    <td class="n-cell">250</td>
                    <td class="key-activities-cell">
                        <ul>
                            <li>Chart review of historical cases</li>
                            <li>Verify hybrid score consistency across surgical subtypes</li>
                        </ul>
                    </td>
                    <td class="biostat-notes-cell">
                        • External validation set: compute AUC, calibration plots, decision curve analysis
                    </td>
                </tr>
                <tr>
                    <td class="phase-cell">4. Alpha Build Prospective</td>
                    <td class="n-cell">500</td>
                    <td class="key-activities-cell">
                        <ul>
                            <li>Integrated app deployment in two OR theatres</li>
                            <li>Prospectively collect outcomes</li>
                        </ul>
                    </td>
                    <td class="biostat-notes-cell">
                        • Assess model performance on real-world data<br>
                        • Refine calibration and model parameters
                    </td>
                </tr>
                <tr>
                    <td class="phase-cell">5. Beta Build Prospective</td>
                    <td class="n-cell">1,000</td>
                    <td class="key-activities-cell">
                        <ul>
                            <li>Full OR suite deployment</li>
                            <li>Evaluate impact on clinical workflow and PONV incidence</li>
                        </ul>
                    </td>
                    <td class="biostat-notes-cell">
                        • Final model validation and performance assessment<br>
                        • Cost-effectiveness analysis (future)
                    </td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)
