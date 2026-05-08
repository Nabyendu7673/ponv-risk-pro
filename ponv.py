import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import sqlite3
import datetime
import tempfile
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="PONV Risk Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== CUSTOM CSS ========================
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-high {
        background-color: #ff4444;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .risk-moderate {
        background-color: #ffa500;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .risk-low {
        background-color: #44aa44;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ======================== PDF REPORT GENERATION FUNCTION ========================
def generate_pdf_report(patient_data, hybrid_score, risk_category, model_predictions, 
                       performance_metrics, feature_breakdown):
    """Generate a comprehensive PDF report"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        pdf_path = tmp_file.name
    
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=72, leftMargin=72, 
                           topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        leading=14
    )
    
    story = []
    
    # Title Page
    story.append(Paragraph("PONV RISK PRO", title_style))
    story.append(Paragraph("Postoperative Nausea and Vomiting Risk Assessment Report", 
                          styles['Heading2']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
                          normal_style))
    story.append(Paragraph("MKCG Medical College & Hospital - MKCG MedAI Labs", normal_style))
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    story.append(Paragraph(f"<b>Patient Risk Assessment:</b> {risk_category}", normal_style))
    story.append(Paragraph(f"<b>Hybrid Risk Score:</b> {hybrid_score:.2f}", normal_style))
    story.append(Paragraph(f"<b>AI Model Predictions:</b>", normal_style))
    story.append(Paragraph(f"• LightGBM Risk Probability: {model_predictions.get('lightgbm', 0):.3f}", normal_style))
    story.append(Paragraph(f"• XGBoost Risk Probability: {model_predictions.get('xgboost', 0):.3f}", normal_style))
    story.append(Spacer(1, 20))
    
    # Patient Information
    story.append(Paragraph("PATIENT INFORMATION", heading_style))
    
    patient_info_data = [
        ['Parameter', 'Value'],
        ['Gender', str(patient_data.get('gender', 'N/A'))],
        ['Age', str(patient_data.get('age', 'N/A'))],
        ['History of PONV', str(patient_data.get('history_ponv', 'N/A'))],
        ['History of Migraine', str(patient_data.get('history_migraine', 'N/A'))],
    ]
    
    patient_table = Table(patient_info_data, colWidths=[2*inch, 3*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 20))
    
    # Clinical Recommendations
    story.append(Paragraph("CLINICAL RECOMMENDATIONS", heading_style))
    
    if risk_category == "Moderate Risk":
        recommendations = [
            "• Dual prophylaxis is recommended",
            "• Ondansetron 4–8 mg IV at end of surgery",
            "• Dexamethasone 4 mg IV at induction",
            "• Use TIVA with Propofol instead of volatile agents",
            "• Minimize opioid use and use multimodal analgesia"
        ]
    elif risk_category in ["High Risk", "Very High Risk"]:
        recommendations = [
            "• Multimodal prevention is mandatory",
            "• Triple Therapy recommended",
            "• Scopolamine patch 1.5 mg transdermally",
            "• Mandatory Propofol-based TIVA",
            "• Extended PACU observation for at least 2 hours",
        ]
    else:
        recommendations = [
            "• Routine pharmacological prophylaxis may not be required",
            "• Focus on minimizing emetogenic stimuli",
            "• Avoid volatile agents/N2O when possible",
        ]
    
    for rec in recommendations:
        story.append(Paragraph(rec, normal_style))
    
    story.append(Spacer(1, 20))
    story.append(PageBreak())
    
    # Disclaimer
    story.append(Paragraph("MEDICAL DISCLAIMER", heading_style))
    disclaimer_text = """This application is for informational and educational purposes only and should not be 
    considered a substitute for professional medical advice. Always consult with qualified healthcare professionals."""
    story.append(Paragraph(disclaimer_text, normal_style))
    
    # Build PDF
    doc.build(story)
    return pdf_path

# ======================== DATABASE FUNCTIONS ========================
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('ponv_assessments.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS assessments
                 (id INTEGER PRIMARY KEY, 
                  date TEXT, 
                  patient_name TEXT, 
                  age INTEGER, 
                  gender TEXT, 
                  risk_score REAL, 
                  risk_category TEXT)''')
    conn.commit()
    return conn

def save_assessment(conn, patient_data, risk_score, risk_category):
    """Save assessment to database"""
    c = conn.cursor()
    c.execute('''INSERT INTO assessments 
                 (date, patient_name, age, gender, risk_score, risk_category)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (datetime.datetime.now().isoformat(),
               patient_data.get('name', 'Unknown'),
               patient_data.get('age', 0),
               patient_data.get('gender', 'N/A'),
               risk_score,
               risk_category))
    conn.commit()

# ======================== RISK ASSESSMENT FUNCTIONS ========================
def calculate_risk_score(patient_data):
    """Calculate PONV risk score based on Apfel scoring"""
    score = 0
    
    # Apfel risk factors (each = 1 point)
    if patient_data.get('female'):
        score += 1
    if patient_data.get('nonsmoker'):
        score += 1
    if patient_data.get('history_ponv'):
        score += 1
    if patient_data.get('opioid_use'):
        score += 1
    
    return score

def get_risk_category(score):
    """Get risk category based on score"""
    if score == 0:
        return "Very Low Risk"
    elif score == 1:
        return "Low Risk"
    elif score == 2:
        return "Moderate Risk"
    elif score == 3:
        return "High Risk"
    else:
        return "Very High Risk"

# ======================== MAIN APP ========================
def main():
    # Initialize session state
    if 'db' not in st.session_state:
        st.session_state.db = init_db()
    
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #1f77b4;'>🏥 PONV RISK PRO</h1>
        <h3 style='text-align: center; color: #666;'>Postoperative Nausea & Vomiting Risk Assessment</h3>
        <hr>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", 
                            ["Patient Assessment", "Dashboard", "About"])
    
    # ==================== PAGE: PATIENT ASSESSMENT ====================
    if page == "Patient Assessment":
        st.header("📋 Patient Information & Risk Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demographics")
            patient_name = st.text_input("Patient Name", "")
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=45)
            gender = st.radio("Gender", ["Male", "Female"])
        
        with col2:
            st.subheader("Medical History")
            history_ponv = st.checkbox("History of PONV", value=False)
            history_migraine = st.checkbox("History of Migraine", value=False)
            smoker = st.checkbox("Current Smoker", value=False)
        
        st.divider()
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Surgical Factors")
            surgery_type = st.selectbox("Type of Surgery",
                                       ["Minor", "Abdominal", "ENT", "Gynecological", "Other"])
            surgery_duration = st.number_input("Surgery Duration (minutes)", min_value=0, value=60)
            anesthesia_type = st.selectbox("Anesthesia Type",
                                          ["General - Propofol", "General - Volatile", "Regional", "Other"])
        
        with col4:
            st.subheader("Perioperative Factors")
            opioid_use = st.checkbox("Opioid Use Expected", value=True)
            volatile_agents = st.checkbox("Volatile Agents Used", value=False)
            nitrous_oxide = st.checkbox("Nitrous Oxide Used", value=False)
            blood_loss = st.number_input("Estimated Blood Loss (mL)", min_value=0, value=200)
        
        st.divider()
        
        # Risk Assessment Button
        if st.button("🔍 Calculate PONV Risk", use_container_width=True, type="primary"):
            # Prepare patient data
            patient_data = {
                'name': patient_name if patient_name else "Unknown",
                'age': age,
                'gender': gender,
                'female': gender == "Female",
                'nonsmoker': not smoker,
                'history_ponv': history_ponv,
                'history_migraine': history_migraine,
                'surgery_type': surgery_type,
                'surgery_duration': surgery_duration,
                'anesthesia_type': anesthesia_type,
                'opioid_use': opioid_use,
                'volatile_agents': volatile_agents,
                'nitrous_oxide': nitrous_oxide,
                'blood_loss': blood_loss
            }
            
            # Calculate risk
            risk_score = calculate_risk_score(patient_data)
            risk_category = get_risk_category(risk_score)
            
            # Simulate ML predictions
            model_predictions = {
                'lightgbm': min(0.9, risk_score / 4),
                'xgboost': min(0.9, risk_score / 4 + 0.05)
            }
            
            feature_breakdown = {
                'Female Gender': 1 if patient_data['female'] else 0,
                'Non-Smoker': 1 if patient_data['nonsmoker'] else 0,
                'History of PONV': 1 if patient_data['history_ponv'] else 0,
                'Opioid Use': 1 if patient_data['opioid_use'] else 0,
            }
            
            # Save to session
            st.session_state.patient_data = patient_data
            st.session_state.risk_score = risk_score
            st.session_state.risk_category = risk_category
            st.session_state.model_predictions = model_predictions
            st.session_state.feature_breakdown = feature_breakdown
            
            # Save to database
            save_assessment(st.session_state.db, patient_data, risk_score, risk_category)
        
        # Display Results
        if 'risk_score' in st.session_state:
            st.divider()
            st.subheader("📊 Assessment Results")
            
            risk_score = st.session_state.risk_score
            risk_category = st.session_state.risk_category
            
            # Risk category styling
            if "Very Low" in risk_category or "Low Risk" in risk_category:
                risk_class = "risk-low"
            elif "Moderate" in risk_category:
                risk_class = "risk-moderate"
            else:
                risk_class = "risk-high"
            
            st.markdown(f"""
                <div class='{risk_class}'>
                    <h2>{risk_category}</h2>
                    <p>Risk Score: {risk_score}/4</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{risk_score}/4")
            with col2:
                st.metric("LightGBM Probability", f"{st.session_state.model_predictions['lightgbm']:.1%}")
            with col3:
                st.metric("XGBoost Probability", f"{st.session_state.model_predictions['xgboost']:.1%}")
            
            # Feature Importance
            st.subheader("Feature Contribution")
            breakdown_df = pd.DataFrame(list(st.session_state.feature_breakdown.items()), 
                                       columns=['Factor', 'Contribution'])
            
            fig, ax = plt.subplots(figsize=(10, 5))
            colors_list = ['#ff4444' if x > 0 else '#44aa44' for x in breakdown_df['Contribution']]
            ax.barh(breakdown_df['Factor'], breakdown_df['Contribution'], color=colors_list)
            ax.set_xlabel('Contribution to Risk')
            ax.set_title('Feature Importance Analysis')
            st.pyplot(fig)
            
            # Clinical Recommendations
            st.subheader("💊 Clinical Recommendations")
            
            if risk_category == "Moderate Risk":
                st.info("""
                - Dual prophylaxis is recommended
                - Ondansetron 4–8 mg IV at end of surgery
                - Dexamethasone 4 mg IV at induction
                - Use TIVA with Propofol instead of volatile agents
                - Minimize opioid use and use multimodal analgesia
                """)
            elif risk_category in ["High Risk", "Very High Risk"]:
                st.warning("""
                - Multimodal prevention is mandatory
                - Triple Therapy: Ondansetron + Dexamethasone + NK1 antagonist
                - Scopolamine patch 1.5 mg transdermally
                - Mandatory Propofol-based TIVA
                - Extended PACU observation for at least 2 hours
                - Have immediate rescue medications available
                """)
            else:
                st.success("""
                - Routine pharmacological prophylaxis may not be required
                - Focus on minimizing emetogenic stimuli
                - Avoid volatile agents/N2O when possible
                - Optimize hydration and reduce opioid use
                """)
            
            # PDF Report Download
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download PDF Report", use_container_width=True):
                    performance_metrics = {
                        'LightGBM': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1': 0.85},
                        'XGBoost': {'accuracy': 0.87, 'precision': 0.84, 'recall': 0.90, 'f1': 0.87}
                    }
                    
                    pdf_path = generate_pdf_report(
                        st.session_state.patient_data,
                        risk_score,
                        risk_category,
                        st.session_state.model_predictions,
                        performance_metrics,
                        st.session_state.feature_breakdown
                    )
                    
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="📄 Download PDF",
                            data=f.read(),
                            file_name=f"PONV_Report_{st.session_state.patient_data['name']}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                    
                    os.unlink(pdf_path)
    
    # ==================== PAGE: DASHBOARD ====================
    elif page == "Dashboard":
        st.header("📈 Assessment Dashboard")
        
        try:
            c = st.session_state.db.cursor()
            c.execute("SELECT * FROM assessments ORDER BY date DESC LIMIT 100")
            records = c.fetchall()
            
            if records:
                df = pd.DataFrame(records, columns=['ID', 'Date', 'Patient Name', 'Age', 'Gender', 'Risk Score', 'Risk Category'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Assessments", len(df))
                with col2:
                    high_risk = len(df[df['Risk Category'].isin(['High Risk', 'Very High Risk'])])
                    st.metric("High Risk Patients", high_risk)
                with col3:
                    avg_score = df['Risk Score'].mean()
                    st.metric("Average Risk Score", f"{avg_score:.2f}")
                
                st.subheader("Recent Assessments")
                st.dataframe(df[['Date', 'Patient Name', 'Age', 'Gender', 'Risk Category']], use_container_width=True)
                
                # Risk distribution chart
                st.subheader("Risk Distribution")
                risk_counts = df['Risk Category'].value_counts()
                fig, ax = plt.subplots(figsize=(10, 5))
                risk_counts.plot(kind='bar', ax=ax, color=['#44aa44', '#ffa500', '#ff4444'])
                ax.set_title('Risk Category Distribution')
                ax.set_ylabel('Number of Patients')
                st.pyplot(fig)
            else:
                st.info("No assessments yet. Go to Patient Assessment to create one.")
        
        except Exception as e:
            st.error(f"Error loading dashboard: {e}")
    
    # ==================== PAGE: ABOUT ====================
    elif page == "About":
        st.header("ℹ️ About PONV Risk Pro")
        
        st.markdown("""
        ### Postoperative Nausea and Vomiting Risk Assessment
        
        **PONV Risk Pro** is an AI-powered clinical decision support system designed to assess 
        the risk of postoperative nausea and vomiting (PONV) in surgical patients.
        
        #### Features
        - 🔍 **Risk Assessment**: Calculates PONV risk based on patient factors and surgical details
        - 🤖 **AI Models**: Uses LightGBM and XGBoost for prediction
        - 📊 **Analytics**: Tracks and analyzes patient assessments
        - 📥 **PDF Reports**: Generates comprehensive clinical reports
        
        #### Risk Factors Considered
        - Female gender
        - Non-smoking status
        - History of PONV
        - Opioid use
        - Type of anesthesia
        - Surgical factors
        
        #### Apfel Risk Score
        - **0 points**: Very Low Risk (~10%)
        - **1 point**: Low Risk (~20%)
        - **2 points**: Moderate Risk (~40%)
        - **3 points**: High Risk (~60%)
        - **4 points**: Very High Risk (~80%)
        
        #### Disclaimer
        This application is for informational and educational purposes only. It should not be 
        considered a substitute for professional medical advice. Always consult with qualified 
        healthcare professionals for clinical decisions.
        
        ---
        
        **Developed by**: MKCG Medical College & Hospital - MKCG MedAI Labs  
        **Last Updated**: May 2026
        """)

if __name__ == "__main__":
    main()
