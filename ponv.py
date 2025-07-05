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
import streamlit.components.v1 as components # Import components for embedding HTML/JS
import sqlite3 # Import sqlite3 for database operations
import datetime # Import datetime for timestamp
import matplotlib.cm as cm

# Add PDF generation imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import io
import base64
from PIL import Image as PILImage
import tempfile
import os

# ------------------------- PDF REPORT GENERATION FUNCTION -------------------------
def generate_pdf_report(patient_data, hybrid_score, risk_category, model_predictions, 
                       feature_importance_fig, roc_fig_train, roc_fig_val, 
                       performance_metrics, feature_breakdown):
    """
    Generate a comprehensive PDF report with all patient data, graphs, and analysis
    """
    # Create a temporary file for the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        pdf_path = tmp_file.name
    
    # Create the PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=72, leftMargin=72, 
                           topMargin=72, bottomMargin=72)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
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
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=12,
        textColor=colors.darkgreen
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        leading=14
    )
    
    # Start building the story (content)
    story = []
    temp_image_paths = []  # <-- Track temp images for later cleanup
    
    # Title Page
    story.append(Paragraph("PONV RISK PRO", title_style))
    story.append(Paragraph("Postoperative Nausea and Vomiting Risk Assessment Report", 
                          styles['Heading2']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
                          normal_style))
    story.append(Paragraph("An initiative of MKCG Medical College & Hospital - MKCG MedAI Labs", 
                          normal_style))
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    story.append(Paragraph(f"<b>Patient Risk Assessment:</b> {risk_category}", normal_style))
    story.append(Paragraph(f"<b>Hybrid Risk Score:</b> {hybrid_score}", normal_style))
    story.append(Paragraph(f"<b>AI Model Predictions:</b>", normal_style))
    story.append(Paragraph(f"• LightGBM Risk Probability: {model_predictions['lightgbm']:.3f}", normal_style))
    story.append(Paragraph(f"• XGBoost Risk Probability: {model_predictions['xgboost']:.3f}", normal_style))
    story.append(Spacer(1, 20))
    
    # Patient Information
    story.append(Paragraph("PATIENT INFORMATION", heading_style))
    
    # Create patient info table
    patient_info_data = [
        ['Parameter', 'Value'],
        ['Gender', patient_data['gender']],
        ['Age', str(patient_data['age'])],
        ['Smoking Status', 'Non-Smoker' if patient_data['smoker'] == 'Yes' else 'Smoker'],
        ['History of PONV', patient_data['history_ponv']],
        ['Preoperative Anxiety', patient_data['preop_anxiety']],
        ['History of Migraine', patient_data['history_migraine']],
        ['BMI > 30', patient_data['obesity']],
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
    
    # Surgical Information
    story.append(Paragraph("SURGICAL INFORMATION", heading_style))
    
    surgical_info_data = [
        ['Parameter', 'Value'],
        ['Abdominal/Laparoscopic Surgery', patient_data['abdominal_surgery']],
        ['ENT/Neurosurgery/Ophthalmic', patient_data['ent_surgery']],
        ['Gynecological/Breast Surgery', patient_data['gynae_surgery']],
        ['Surgery Duration > 60 min', patient_data['surgery_duration']],
        ['Major Blood Loss > 500 mL', patient_data['major_blood_loss']],
        ['Use of Volatile Agents', patient_data['volatile_agents']],
        ['Use of Nitrous Oxide', patient_data['nitrous_oxide']],
    ]
    
    surgical_table = Table(surgical_info_data, colWidths=[2*inch, 3*inch])
    surgical_table.setStyle(TableStyle([
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
    story.append(surgical_table)
    story.append(Spacer(1, 20))
    
    # Drug Administration
    story.append(Paragraph("DRUG ADMINISTRATION", heading_style))
    
    drug_info_data = [
        ['Drug', 'Dose', 'Route'],
        ['Ondansetron', f"{patient_data['ondansetron_dose']} mg", 'IV'],
        ['Midazolam', f"{patient_data['midazolam_dose']} mg", 'IV'],
        ['Dexamethasone', f"{patient_data['dexamethasone_dose']} mg", 'IV'],
        ['Glycopyrrolate', f"{patient_data['glycopyrrolate_dose']} mg", 'IV'],
        ['Nalbuphine', f"{patient_data['nalbuphine_dose']} mg", 'IV'],
        ['Fentanyl', f"{patient_data['fentanyl_dose']} mcg", 'IV'],
        ['Butorphanol', f"{patient_data['butorphanol_dose']} mg", 'IV'],
        ['Pentazocine', f"{patient_data['pentazocine_dose']} mg", 'IV'],
        ['Propofol Mode', patient_data['propofol_mode'], ''],
        ['Muscle Relaxant', f"{patient_data['muscle_relaxant']} ({patient_data['muscle_relaxant_dose']} mg/kg)", ''],
    ]
    
    drug_table = Table(drug_info_data, colWidths=[1.5*inch, 1.5*inch, 1*inch])
    drug_table.setStyle(TableStyle([
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
    story.append(drug_table)
    story.append(PageBreak())
    
    # Risk Assessment Results
    story.append(Paragraph("RISK ASSESSMENT RESULTS", heading_style))
    
    # Risk category with color coding
    risk_color = {
        "Very Low Risk": colors.green,
        "Low Risk": colors.blue,
        "Moderate Risk": colors.orange,
        "High Risk": colors.red,
        "Very High Risk": colors.darkred
    }
    
    risk_data = [
        ['Assessment Type', 'Result', 'Risk Level'],
        ['Hybrid Risk Score', str(hybrid_score), risk_category],
        ['LightGBM Prediction', f"{model_predictions['lightgbm']:.3f}", 
         "High" if model_predictions['lightgbm'] > 0.5 else "Low"],
        ['XGBoost Prediction', f"{model_predictions['xgboost']:.3f}", 
         "High" if model_predictions['xgboost'] > 0.5 else "Low"],
    ]
    
    risk_table = Table(risk_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('BACKGROUND', (2, 1), (2, 1), risk_color.get(risk_category, colors.grey)),
        ('BACKGROUND', (2, 2), (2, 2), colors.green if model_predictions['lightgbm'] <= 0.5 else colors.red),
        ('BACKGROUND', (2, 3), (2, 3), colors.green if model_predictions['xgboost'] <= 0.5 else colors.red),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 20))
    
    # Feature Importance Plot
    if feature_importance_fig is not None:
        story.append(Paragraph("FEATURE IMPORTANCE ANALYSIS", heading_style))
        story.append(Paragraph("The following chart shows the relative importance of different factors in predicting PONV risk:", normal_style))
        
        # Save the matplotlib figure to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
            feature_importance_fig.savefig(tmp_img.name, dpi=300, bbox_inches='tight', facecolor='white')
            img_path = tmp_img.name
        
        # Add the image to the PDF
        img = Image(img_path, width=6*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 20))
        
        # Defer cleanup until after PDF is built
        temp_image_paths.append(img_path)
    
    # ROC Curves
    if roc_fig_train is not None or roc_fig_val is not None:
        story.append(Paragraph("MODEL PERFORMANCE - ROC CURVES", heading_style))
        story.append(Paragraph("Receiver Operating Characteristic (ROC) curves showing model performance:", normal_style))
        
        # Training ROC
        if roc_fig_train is not None:
            story.append(Paragraph("Training Data ROC Curve:", subheading_style))
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
                roc_fig_train.savefig(tmp_img.name, dpi=300, bbox_inches='tight', facecolor='white')
                img_path = tmp_img.name
            
            img = Image(img_path, width=5*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 10))
            temp_image_paths.append(img_path)
        
        # Validation ROC
        if roc_fig_val is not None:
            story.append(Paragraph("Validation Data ROC Curve:", subheading_style))
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
                roc_fig_val.savefig(tmp_img.name, dpi=300, bbox_inches='tight', facecolor='white')
                img_path = tmp_img.name
            
            img = Image(img_path, width=5*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 20))
            temp_image_paths.append(img_path)
    
    # Performance Metrics
    if performance_metrics is not None:
        story.append(Paragraph("MODEL PERFORMANCE METRICS", heading_style))
        
        # Create performance metrics table
        metrics_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]
        for model_name, metrics in performance_metrics.items():
            metrics_data.append([
                model_name,
                f"{metrics['accuracy']:.3f}",
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1']:.3f}"
            ])
        
        metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))
    
    # Feature Breakdown
    if feature_breakdown is not None:
        story.append(Paragraph("DETAILED SCORING BREAKDOWN", heading_style))
        story.append(Paragraph("Individual contribution of each factor to the risk score:", normal_style))
        
        # Create feature breakdown table
        breakdown_data = [['Factor', 'Contribution', 'Impact']]
        for factor, score in feature_breakdown.items():
            impact = "Risk Increase" if score > 0 else "Risk Decrease" if score < 0 else "Neutral"
            breakdown_data.append([factor, str(score), impact])
        
        breakdown_table = Table(breakdown_data, colWidths=[3*inch, 1*inch, 1.5*inch])
        breakdown_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(breakdown_table)
        story.append(Spacer(1, 20))
    
    # Clinical Recommendations
    story.append(Paragraph("CLINICAL RECOMMENDATIONS", heading_style))
    
    if risk_category == "Moderate Risk":
        recommendations = [
            "• Dual prophylaxis is recommended",
            "• Ondansetron 4–8 mg IV at end of surgery",
            "• Dexamethasone 4 mg IV at induction",
            "• Consider Midazolam 1–2 mg IV if preoperative anxiety present",
            "• Use TIVA with Propofol instead of volatile agents",
            "• Minimize opioid use and use multimodal analgesia",
            "• Avoid Nitrous Oxide",
            "• Ensure adequate hydration and gastric decompression",
            "• Monitor in PACU for >30 minutes"
        ]
    elif risk_category in ["High Risk", "Very High Risk"]:
        recommendations = [
            "• Multimodal prevention is mandatory",
            "• Triple Therapy: Ondansetron 4–8 mg IV + Dexamethasone 4–8 mg IV + NK1 receptor antagonist",
            "• Scopolamine patch 1.5 mg transdermally",
            "• Consider Droperidol 0.625–1.25 mg IV if QT prolongation not present",
            "• Mandatory Propofol-based TIVA",
            "• Use opioid-sparing strategies with nerve blocks or adjuncts",
            "• Avoid volatile agents and N2O unless absolutely necessary",
            "• Extended PACU observation for at least 2 hours",
            "• Have immediate rescue medications available",
            "• Provide discharge prescription for anti-emetics"
        ]
    else:  # Low or Very Low Risk
        recommendations = [
            "• Routine pharmacological prophylaxis may not be required",
            "• Focus on minimizing emetogenic stimuli",
            "• Avoid volatile agents/N2O when possible",
            "• Consider regional techniques",
            "• Optimize hydration and reduce opioid use"
        ]
    
    for rec in recommendations:
        story.append(Paragraph(rec, normal_style))
    
    story.append(Spacer(1, 20))
    
    # Rescue Therapy
    story.append(Paragraph("RESCUE THERAPY (for breakthrough PONV)", subheading_style))
    rescue_therapy = [
        "First-line Rescue:",
        "• Metoclopramide 10 mg IV",
        "• Promethazine 12.5–25 mg IV",
        "",
        "Second-line Rescue:",
        "• Scopolamine patch (if not previously used)",
        "• Haloperidol 0.5–1 mg IV (if QTc is normal)",
        "",
        "Important: Do not repeat the same class used for prophylaxis"
    ]
    
    for therapy in rescue_therapy:
        if therapy.startswith("•"):
            story.append(Paragraph(therapy, normal_style))
        elif therapy == "":
            story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(therapy, subheading_style))
    
    story.append(PageBreak())
    
    # Disclaimer and References
    story.append(Paragraph("MEDICAL DISCLAIMER", heading_style))
    disclaimer_text = """
    This application is for informational and educational purposes only and should not be considered a substitute for professional medical advice. The predictions and recommendations provided are based on statistical models and should be used as decision support tools only. Always consult with a qualified healthcare provider for diagnosis and treatment decisions.
    
    Developed by: MKCG Medical College & Hospital - MKCG MedAI Labs
    Last Updated: December 2024
    """
    story.append(Paragraph(disclaimer_text, normal_style))
    
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("REFERENCES", heading_style))
    references = [
        "1. Apfel CC, et al. A simplified risk score for predicting postoperative nausea and vomiting. Anesthesiology 1999;91:693–700.",
        "2. Koivuranta M, et al. A survey of postoperative nausea and vomiting. Anaesthesia 1997;52:443–449.",
        "3. Fourth Consensus Guidelines for the Management of Postoperative Nausea and Vomiting. Anesth Analg 2020;131:411-448.",
        "4. Gan TJ, et al. Consensus guidelines for the management of postoperative nausea and vomiting. Anesth Analg 2014;118:85-113."
    ]
    
    for ref in references:
        story.append(Paragraph(ref, normal_style))
    
    # Build the PDF
    doc.build(story)
    
    # Clean up all temp images after PDF is built
    for img_path in temp_image_paths:
        try:
            os.unlink(img_path)
        except Exception:
            pass
    
    return pdf_path

# ------------------------- END PDF REPORT GENERATION FUNCTION -------------------------



# Core Setup and UI
st.set_page_config(layout="wide")

# Inject custom CSS for styling the app and the flowchart
# This block contains all the CSS rules, including those for the flowchart
st.markdown("""
<style>
body {
    color: #ffffff !important;
    font-family: 'Inter', sans-serif;
}
.main .block-container {
    color: #ffffff !important;
    border-radius: 16px;
    box-shadow: 0 4px 32px rgba(0,0,0,0.25);
    padding: 2.5rem 2rem;
    margin-top: 2rem;
}
.stSidebar, .css-1d391kg, .css-1lcbmhc {
    color: #ffffff !important;
}
.stButton > button {
    background: linear-gradient(90deg, #ffb366 0%, #ff8800 100%);
    color: #18191a;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    transition: 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #ff8800 0%, #ffb366 100%);
    color: #fff;
}
.animated-title {
    font-size: 4em;
    font-weight: 800;
    text-align: center;
    color: #ffffff !important;
    text-shadow: 0 2px 16px #ff8800, 0 0px 2px #fff;
    letter-spacing: -1px;
    margin-bottom: 0.2em;
    margin-top: 0.2em;
}
@keyframes gradientMove {
    0% {background-position:0% 50%}
    50% {background-position:100% 50%}
    100% {background-position:0% 50%}
}
.card, .hybrid-score-box, .dose-box, .streamlit-expander {
    color: #ffffff !important;
    border-radius: 12px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.25);
}
table {
    color: #ffffff !important;
}
th, td {
    border-color: #444 !important;
    color: #ffffff !important;
}
tr:nth-child(even) {
    background-color: #202124 !important;
}
tr:hover td {
    background-color: #333 !important;
    color: #ffffff !important;
}
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #444;
    border-radius: 4px;
}

/* Import professional fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* General Body Styling */
body {
    font-family: 'Inter', sans-serif;
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
    font-size: 6em;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(120deg, #2b5876, #4e4376);
    -webkit-background-clip: text;
    -webkit-text-fill-color: #000000;
}

/* Sidebar Enhancement */
.stSidebar {
    padding: 2rem;
    border-right: 1px solid rgba(0,0,0,0.1);
    box-shadow: 2px 0 8px rgba(0,0,0,0.05);
}

/* Input Fields Enhancement */
.stSelectbox div[data-baseweb="select"],
.stNumberInput div[data-baseweb="input"] {
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
    background: #2ecc40 !important;
    color: #fff !important;
}
.low-risk {
    background: #3498db !important;
    color: #fff !important;
}
.moderate-risk {
    background: #ffe066 !important;
    color: #18191a !important;
}
.high-risk {
    background: #e74c3c !important;
    color: #fff !important;
}
.very-high-risk {
    background: #636363 !important;
    color: #fff !important;
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
    border: 1px solid #e0e0e0;
}

/* Table Header Enhancement */
th {
    background: #18191a !important;
    color: #ffffff !important;
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
    transition: all 0.3s ease;
}

.streamlit-expander:hover {
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
}

/* Dose Box Enhancement */
.dose-box {
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    border: 1px solid #e0e0e0;
}

.dose-info {
    color: #ffffff !important;
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

/* Loading Animation */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #ff8800;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-right: 10px;
}

.loading-text {
    animation: pulse 2s infinite;
    color: #ff8800;
    font-weight: 600;
}

/* Progress Bar Enhancement */
.stProgress > div > div > div > div {
    background-color: #ff8800 !important;
}

/* Success/Error Message Animations */
@keyframes slideIn {
    from { transform: translateY(-20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

.success-message {
    animation: slideIn 0.5s ease-out;
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border-left: 5px solid #28a745;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}

.error-message {
    animation: slideIn 0.5s ease-out;
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border-left: 5px solid #dc3545;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}

/* Card Hover Effects */
.interactive-card {
    transition: all 0.3s ease;
    cursor: pointer;
    border: 2px solid transparent;
}

.interactive-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    border-color: #ff8800;
}

/* Gradient Text Animation */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.gradient-text {
    background: linear-gradient(-45deg, #ff8800, #ffb366, #ff6b35, #ff8800);
    background-size: 400% 400%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 3s ease infinite;
    font-weight: 700;
}

/* Floating Action Button */
.fab {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 60px;
    height: 60px;
    background: linear-gradient(135deg, #ff8800, #ffb366);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 24px;
    box-shadow: 0 4px 15px rgba(255, 136, 0, 0.4);
    transition: all 0.3s ease;
    z-index: 1000;
}

.fab:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 20px rgba(255, 136, 0, 0.6);
}

/* Tooltip Enhancement */
.tooltip {
    position: relative;
    display: inline-block;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: #333;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 12px;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}

/* Sidebar Section Headers */
.sidebar-section {
    background: linear-gradient(135deg, #2b5876, #4e4376);
    color: white;
    padding: 10px 15px;
    border-radius: 8px;
    margin: 15px 0 10px 0;
    font-weight: 600;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Input Field Focus Effects */
.stSelectbox div[data-baseweb="select"]:focus-within,
.stNumberInput div[data-baseweb="input"]:focus-within {
    border-color: #ff8800 !important;
    box-shadow: 0 0 0 3px rgba(255, 136, 0, 0.2) !important;
    transform: scale(1.02);
}

/* Risk Score Visualization */
.risk-meter {
    width: 100%;
    height: 20px;
    background: linear-gradient(90deg, #2ecc40 0%, #3498db 25%, #ffe066 50%, #e74c3c 75%, #636363 100%);
    border-radius: 10px;
    position: relative;
    margin: 10px 0;
    overflow: hidden;
}

.risk-indicator {
    position: absolute;
    top: -2px;
    width: 4px;
    height: 24px;
    background: #fff;
    border-radius: 2px;
    box-shadow: 0 0 5px rgba(0,0,0,0.3);
    transition: left 0.5s ease;
}

/* Notification Badge */
.notification-badge {
    background: #e74c3c;
    color: white;
    border-radius: 50%;
    padding: 2px 6px;
    font-size: 10px;
    position: absolute;
    top: -5px;
    right: -5px;
    min-width: 15px;
    text-align: center;
}

/* Responsive Design Improvements */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem !important;
        margin-top: 1rem !important;
    }
    
    .animated-title {
        font-size: 2.5em !important;
    }
    
    .sidebar-section {
        font-size: 0.9em;
        padding: 8px 12px;
    }
}

/* Dark Mode Toggle */
.dark-mode-toggle {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
}

/* Data Visualization Enhancements */
.chart-container {
    border-radius: 12px;
    padding: 20px;
    margin: 15px 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Status Indicators */
.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-online { background: #2ecc40; }
.status-warning { background: #f39c12; }
.status-error { background: #e74c3c; }

/* Enhanced Buttons */
.btn-primary {
    background: linear-gradient(135deg, #ff8800, #ffb366);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
    cursor: pointer;
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(255, 136, 0, 0.4);
}

.btn-secondary {
    background: linear-gradient(135deg, #6c757d, #495057);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.3s ease;
    cursor: pointer;
}

.btn-secondary:hover {
    transform: translateY(-1px);
    box-shadow: 0 3px 10px rgba(108, 117, 125, 0.4);
}
</style>
""", unsafe_allow_html=True)

# Add tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Main Interface",
    "Detailed Scoring Guide",
    "Model Training Timeline and Methodological Summary",
    "References",
    "Global Feature Importance"
])

with tab1:
    st.markdown("""
    <div style='font-size:3em; font-weight:800; color:#fff; text-align:center; margin-bottom:0.2em; margin-top:0.2em;'>PONV RISK PRO</div>
    """, unsafe_allow_html=True)

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

    # Enhanced Sidebar with Section Headers
    st.sidebar.markdown("""
    <div class='sidebar-section'>
        <i class='fas fa-user-md'></i> PONV Risk Assessment Parameters
    </div>
    """, unsafe_allow_html=True)

    # ------------------------- PATIENT FACTORS -------------------------
    st.sidebar.markdown("""
    <div style='font-weight: 600; color: #ff8800; margin: 15px 0 10px 0; border-bottom: 2px solid #ff8800; padding-bottom: 5px;'>
        👤 Patient Factors
    </div>
    """, unsafe_allow_html=True)
    
    gender = st.sidebar.selectbox("Female Gender", ["No", "Yes"], help="Female patients have higher PONV risk")
    smoker = st.sidebar.selectbox("Non-Smoker", ["No", "Yes"], help="Non-smokers have higher PONV risk")
    history_ponv = st.sidebar.selectbox("History of PONV or Motion Sickness", ["No", "Yes"], help="Previous PONV episodes increase risk")
    age = st.sidebar.slider("Age", 18, 80, 35, help="Age > 50 years increases PONV risk")
    preop_anxiety = st.sidebar.selectbox("Preoperative Anxiety", ["No", "Yes"], help="Anxiety can increase PONV risk")
    history_migraine = st.sidebar.selectbox("History of Migraine", ["No", "Yes"], help="Migraine history correlates with PONV")
    obesity = st.sidebar.selectbox("BMI > 30", ["No", "Yes"], help="Obesity is a risk factor for PONV")

    # ------------------------- SURGICAL FACTORS -------------------------
    st.sidebar.markdown("""
    <div style='font-weight: 600; color: #ff8800; margin: 15px 0 10px 0; border-bottom: 2px solid #ff8800; padding-bottom: 5px;'>
        🏥 Surgical Factors
    </div>
    """, unsafe_allow_html=True)
    
    abdominal_surgery = st.sidebar.selectbox("Abdominal or Laparoscopic Surgery", ["No", "Yes"], help="Laparoscopic procedures increase PONV risk")
    ent_surgery = st.sidebar.selectbox("ENT/Neurosurgery/Ophthalmic Surgery", ["No", "Yes"], help="ENT and neurosurgical procedures are high-risk")
    gynae_surgery = st.sidebar.selectbox("Gynecological or Breast Surgery", ["No", "Yes"], help="Gynecological procedures have higher PONV incidence")
    surgery_duration = st.sidebar.selectbox("Surgery Duration > 60 min", ["No", "Yes"], help="Longer procedures increase PONV risk")
    major_blood_loss = st.sidebar.selectbox("Major Blood Loss > 500 mL", ["No", "Yes"], help="Significant blood loss can trigger PONV")
    volatile_agents = st.sidebar.selectbox("Use of Volatile Agents (Sevo/Iso/Des)", ["No", "Yes"], help="Volatile anesthetics are emetogenic")
    nitrous_oxide = st.sidebar.selectbox("Use of Nitrous Oxide", ["No", "Yes"], help="N2O increases PONV risk")

    # ------------------------- DRUG FACTORS (WITH DOSE) -------------------------
    st.sidebar.markdown("""
    <div style='font-weight: 600; color: #ff8800; margin: 15px 0 10px 0; border-bottom: 2px solid #ff8800; padding-bottom: 5px;'>
        💊 Drug Administration (Specify Dose)
    </div>
    """, unsafe_allow_html=True)

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

    # Add muscle relaxant dose input in the sidebar
    muscle_relaxant_dose = st.sidebar.number_input("Muscle Relaxant Dose (mg/kg)", 0.0, 5.0, 0.0, key='muscle_relaxant_dose')

    # ------------------------- HYBRID SCORING FUNCTION -------------------------
    def binary(val):
        return 1 if val == "Yes" else 0

    def propofol_score(mode):
        return -3 if mode == "TIVA" else -1 if mode == "Induction Only" else 0

    def midazolam_score(dose):
        if dose == 0:
            return 0
        elif dose <= 2:
            return -1
        elif dose <= 10:
            return -2
        else:
            return -3

    def ondansetron_score(dose):
        if dose == 0:
            return 0
        elif dose < 4:
            return -1
        elif dose < 8:
            return -2
        else:
            return -3

    def dexamethasone_score(dose):
        if dose == 0:
            return 0
        elif dose < 4:
            return -1
        elif dose <= 10:
            return -2
        else:
            return -3

    def glycopyrrolate_score(dose):
        if dose == 0:
            return 0
        elif dose <= 0.2:
            return 1
        else:
            return 2

    def nalbuphine_score(dose):
        if dose == 0:
            return 0
        elif dose <= 10:
            return 1
        else:
            return 2

    def fentanyl_score(dose):
        if dose == 0:
            return 0
        elif dose <= 100:
            return 1
        elif dose <= 500:
            return 2
        else:
            return 3

    def butorphanol_score(dose):
        if dose == 0:
            return 0
        elif dose <= 2:
            return 1
        else:
            return 2

    def pentazocine_score(dose):
        if dose == 0:
            return 0
        elif dose <= 100:
            return 1
        elif dose <= 200:
            return 2
        else:
            return 3

    def muscle_relaxant_score(muscle_relaxant, dose):
        if muscle_relaxant == "None":
            return 0
        elif muscle_relaxant == "Succinylcholine":
            if dose < 1.5:
                return 1
            else:
                return 2
        elif muscle_relaxant == "Rocuronium":
            if dose < 0.6:
                return 0
            elif dose <= 1.0:
                return 1
            else:
                return 2
        elif muscle_relaxant == "Vecuronium":
            if dose < 0.1:
                return 0
            else:
                return 1
        elif muscle_relaxant == "Atracurium" or muscle_relaxant == "Cisatracurium":
            if dose < 0.4:
                return 1
            else:
                return 2
        else:
            return 0

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
        if abdominal_surgery == "Yes":
            score += 1
        if ent_surgery == "Yes":
            score += 1
        if gynae_surgery == "Yes":
            score += 1
        if surgery_duration == "Yes":
            score += 1
        if major_blood_loss == "Yes":
            score += 1
        if volatile_agents == "Yes":
            score += 1
        if nitrous_oxide == "Yes":
            score += 1
        # Dose-dependent drug scores
        score += midazolam_score(midazolam_dose)
        score += ondansetron_score(ondansetron_dose)
        score += dexamethasone_score(dexamethasone_dose)
        score += glycopyrrolate_score(glycopyrrolate_dose)
        score += nalbuphine_score(nalbuphine_dose)
        score += fentanyl_score(fentanyl_dose)
        score += butorphanol_score(butorphanol_dose)
        score += pentazocine_score(pentazocine_dose)
        score += propofol_score(propofol_mode)
        score += muscle_relaxant_score(muscle_relaxant, muscle_relaxant_dose)
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

    # Calculate position for risk meter (0-100%)
    def get_risk_percentage(score):
        if score <= -5:
            return 5  # Very Low Risk
        elif -4 <= score <= 3:
            return 25  # Low Risk
        elif 4 <= score <= 9:
            return 50  # Moderate Risk
        elif 10 <= score <= 15:
            return 75  # High Risk
        else:
            return 95  # Very High Risk

    risk_percentage = get_risk_percentage(hybrid_score)
    total_score = hybrid_score
    risk_category_label = category

    # Display the risk meter
    st.markdown(f"""
<div class='risk-meter-container'>
    <h2 class='total-score'>Total Hybrid Score: {total_score}</h2>
    <h3 class='risk-category'>Risk Category: {risk_category_label}</h3>
    <div class='risk-meter'>
        <div class='risk-indicator' style='left: {risk_percentage}%;'></div>
    </div>
    <div class='risk-labels'>
        <span>Very Low</span>
        <span>Low</span>
        <span>Moderate</span>
        <span>High</span>
        <span>Very High</span>
    </div>
</div>
""", unsafe_allow_html=True)

    # Add CSS for the risk meter with black background and visible text
    st.markdown("""
<style>
.risk-meter-container {
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.4);
}
.total-score {
    color: #ffb366;
    font-size: 24px;
    margin-bottom: 10px;
}
.risk-category {
    color: #fff;
    font-size: 20px;
    margin-bottom: 20px;
}
.risk-meter {
    height: 20px;
    background: linear-gradient(to right, #4daf4a, #ffed4a, #ff4444);
    border-radius: 10px;
    position: relative;
    margin-bottom: 10px;
}
.risk-indicator {
    width: 12px;
    height: 30px;
    background: #fff;
    position: absolute;
    top: -5px;
    transform: translateX(-50%);
    border-radius: 3px;
}
.risk-labels {
    display: flex;
    justify-content: space-between;
    color: #fff;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


    # ------------------------- RECOMMENDATIONS FOR CLINICIANS (Expandable) -------------------------

    st.subheader("Recommendations for Clinicians")
    with st.expander("View Recommendations"):
        if category == "Moderate Risk":
            st.markdown("""
            <h4>🔹 1. Moderate Risk (Score: 4–9)</h4>
            <b>Moderate Risk (Score: 4–9)</b><br>
            <span style='color:#007bff;'>&#x27A1; Dual prophylaxis is recommended</span><br>
            <span style='font-size:0.95em;'>📘 Source: <a href="https://pubmed.ncbi.nlm.nih.gov/32049718/" target="_blank">Fourth Consensus Guidelines, 2020 (ASHP)</a></span>
            <ul>
                <li><b>Ondansetron 4–8 mg IV</b> (5-HT<sub>3</sub> antagonist) – Administer at end of surgery</li>
                <li><b>Dexamethasone 4 mg IV</b> – At induction (slow onset)</li>
                <li><b>Midazolam 1–2 mg IV</b> – Consider if preoperative anxiety present</li>
            </ul>
            <b>Anesthetic Techniques:</b>
            <ul>
                <li><b>TIVA (Total IV Anesthesia) with Propofol</b> – Preferred over volatile agents</li>
                <li><b>Minimize opioid use</b> – Use multimodal analgesia (e.g., NSAIDs, acetaminophen)</li>
                <li><b>Avoid Nitrous Oxide (N<sub>2</sub>O)</b> – Linked to increased PONV</li>
            </ul>
            <b>Supportive Measures:</b>
            <ul>
                <li><b>Adequate hydration</b> – To reduce nausea from hypovolemia</li>
                <li><b>Gastric decompression</b> – Avoid distension</li>
                <li><b>Observation &gt;30 min</b> – Monitor in PACU</li>
            </ul>
            """, unsafe_allow_html=True)
        elif category in ["High Risk", "Very High Risk"]:
            st.markdown("""
            <h4>🔴 2. High Risk (Score: 10–15) or Very High Risk</h4>
            <b>High Risk (Score: 10–15) or Very High Risk</b><br>
            <span style='color:#dc3545;'>&#x27A1; Multimodal prevention is mandatory</span><br>
            <span style='font-size:0.95em;'>📘 Source: <a href="https://pubmed.ncbi.nlm.nih.gov/32049718/" target="_blank">2020 PONV Guidelines (SAMBA & ASHP)</a></span><br>
            <span style='font-size:0.95em;'>📘 Also see: <a href="https://www.fda.gov/regulatory-information/search-fda-guidance-documents/postoperative-nausea-and-vomiting-patients-undergoing-surgery-guidance-industry" target="_blank">FDA Draft Guidance on PONV, 2024</a></span>
            <ul>
                <li><b>Triple Therapy:</b>
                    <ul>
                        <li>Ondansetron 4–8 mg IV</li>
                        <li>Dexamethasone 4–8 mg IV</li>
                        <li>NK1 receptor antagonist (e.g., Aprepitant 40 mg PO or Fosaprepitant 150 mg IV)</li>
                    </ul>
                </li>
                <li>Scopolamine patch – 1.5 mg transdermally (apply night before or 2 hrs pre-op)</li>
                <li>Droperidol 0.625–1.25 mg IV – If QT prolongation is not present</li>
            </ul>
            <b>Anesthetic Techniques:</b>
            <ul>
                <li><b>Mandatory Propofol-based TIVA</b> – Eliminates volatile anesthetics</li>
                <li><b>Opioid-sparing strategies</b> – Use nerve blocks or adjuncts like ketamine/dexmedetomidine</li>
                <li><b>Avoid volatile agents &amp; N<sub>2</sub>O</b> – Unless absolutely necessary</li>
            </ul>
            <b>Postoperative Care:</b>
            <ul>
                <li><b>Extended PACU observation</b> – At least 2 hours</li>
                <li><b>Immediate rescue meds available</b></li>
                <li><b>Discharge prescription</b> – Anti-emetics like ondansetron or promethazine</li>
            </ul>
            """, unsafe_allow_html=True)
        elif category in ["Very Low Risk", "Low Risk"]:
            st.markdown("""
            <h4>🟢 3. Very Low and Low Risk (Score: 0–3)</h4>
            <span style='font-size:0.95em;'>📘 Source: <a href="https://www.ashp.org/-/media/assets/policy-guidelines/docs/guidelines/postoperative-nausea-vomiting.ashx" target="_blank">ASHP Guidelines</a></span>
            <ul>
                <li>Routine pharmacological prophylaxis may not be required.</li>
                <li>Focus on minimizing emetogenic stimuli:</li>
                <ul>
                    <li>Avoid volatile agents/N<sub>2</sub>O when possible</li>
                    <li>Consider regional techniques</li>
                    <li>Optimize hydration and reduce opioid use</li>
                </ul>
            </ul>
            """, unsafe_allow_html=True)
        # Rescue Therapy section (always shown)
        st.markdown("""
        <hr>
        <h4>🩺 4. Rescue Therapy (for breakthrough PONV despite prophylaxis)</h4>
        <b>First-line Rescue:</b>
        <ul>
            <li>Metoclopramide 10 mg IV</li>
            <li>Promethazine 12.5–25 mg IV</li>
        </ul>
        <b>Second-line Rescue:</b>
        <ul>
            <li>Scopolamine patch – If not previously used</li>
            <li>Haloperidol 0.5–1 mg IV – Use if QTc is normal</li>
        </ul>
        <b style='color:#dc3545;'>❗ Do not repeat the same class used for prophylaxis (e.g., avoid repeat 5-HT<sub>3</sub> if used already)</b>
        <br>
        <span style='font-size:0.95em;'>📘 Reference: <a href="https://www.openanesthesia.org/po_nausea_vomiting/" target="_blank">OpenAnesthesia: PONV Management</a></span>
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
    # Use 500 synthetic samples for faster demo
    @st.cache_data
    def generate_synthetic_data(n_samples=500, n_features=23):
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples)
        np.random.seed(42)
        for i in range(n_samples):
            X[i, 0:14] = np.random.binomial(1, 0.5, 14)
            X[i, 3] = np.random.normal(45, 15)
            X[i, 14:22] = np.maximum(0, np.random.exponential(2, 8))
            X[i, 22] = np.random.choice([-3, -1, 0])
            risk_factors = (
                2.5 * X[i, 0] +
                2.0 * X[i, 2] +
                1.5 * X[i, 4] +
                1.5 * X[i, 5] +
                1.2 * X[i, 6] +
                2.0 * X[i, 7] +
                1.5 * X[i, 8] +
                1.5 * X[i, 9] +
                1.2 * X[i, 11] +
                2.5 * X[i, 12] +
                1.5 * X[i, 13]
            )
            protective_factors = (
                1.5 * X[i, 14] +
                2.5 * X[i, 15] +
                2.5 * X[i, 16] +
                1.5 * X[i, 17]
            )
            # Adjusted sigmoid for ROC AUC ~0.8-0.9
            prob = 1 / (1 + np.exp(-2.0 * (risk_factors - protective_factors)))
            y[i] = np.random.binomial(1, prob)
        return X, y

    # Generate synthetic data
    n_features = 23
    X, y = generate_synthetic_data(500, n_features)

    # Split and preprocess data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Add feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Add SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    # Cache model training so it only runs when data changes
    @st.cache_resource
    def train_models(X_train_balanced, y_train_balanced):
        xgb_model = XGBClassifier(
            max_depth=3,
            learning_rate=0.03,
            n_estimators=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='auc'
        )
        xgb_model.fit(X_train_balanced, y_train_balanced)
        lgb_model = lgb.LGBMClassifier(n_estimators=50, max_depth=3)
        lgb_model.fit(X_train_balanced, y_train_balanced)
        return xgb_model, lgb_model

    # Train models (cached)
    xgb_model, lgb_model = train_models(X_train_balanced, y_train_balanced)


    


    # ------------------------- MODEL EVALUATION -------------------------
    st.subheader("Model AUC Scores")

    auc_xgb_train, auc_lgb_train = None, None
    auc_xgb_val, auc_lgb_val = None, None
    fig_train_all, fig_val_all = None, None

    if len(np.unique(y_train_balanced)) < 2:
        st.warning("Training data contains only one class. Cannot calculate ROC curves and AUC for training data.")
    else:
        fpr_xgb_train, tpr_xgb_train, _ = roc_curve(y_train_balanced, xgb_model.predict_proba(X_train_balanced)[:, 1])
        fpr_lgb_train, tpr_lgb_train, _ = roc_curve(y_train_balanced, lgb_model.predict_proba(X_train_balanced)[:, 1])
        auc_xgb_train = auc(fpr_xgb_train, tpr_xgb_train)
        auc_lgb_train = auc(fpr_lgb_train, tpr_lgb_train)
        fig_train_all, ax_train_all = plt.subplots(figsize=(5, 3))
        fig_train_all.patch.set_facecolor('#ffffff')
        ax_train_all.set_facecolor('#ffffff')
        ax_train_all.tick_params(colors='#000000')
        ax_train_all.xaxis.label.set_color('#000000')
        ax_train_all.yaxis.label.set_color('#000000')
        ax_train_all.title.set_color('#000000')
        ax_train_all.plot(fpr_lgb_train, tpr_lgb_train, label=f"LightGBM (AUC = {auc_lgb_train:.3f})")
        ax_train_all.plot(fpr_xgb_train, tpr_xgb_train, label=f"XGBoost (AUC = {auc_xgb_train:.3f})")
        ax_train_all.plot([0, 1], [0, 1], 'k--')
        ax_train_all.set_xlabel("False Positive Rate")
        ax_train_all.set_ylabel("True Positive Rate")
        ax_train_all.set_title("Training ROC Curve (LightGBM & XGBoost)")
        ax_train_all.legend(loc="lower right", fontsize='small')


    if len(np.unique(y_val)) < 2:
        st.warning("Validation data contains only one class. Cannot calculate ROC curves and AUC for validation data.")
    else:
        fpr_xgb_val, tpr_xgb_val, _ = roc_curve(y_val, xgb_model.predict_proba(X_val_scaled)[:, 1])
        fpr_lgb_val, tpr_lgb_val, _ = roc_curve(y_val, lgb_model.predict_proba(X_val_scaled)[:, 1])
        auc_xgb_val = auc(fpr_xgb_val, tpr_xgb_val)
        auc_lgb_val = auc(fpr_lgb_val, tpr_lgb_val)
        fig_val_all, ax_val_all = plt.subplots(figsize=(5, 3))
        fig_val_all.patch.set_facecolor('#ffffff')
        ax_val_all.set_facecolor('#ffffff')
        ax_val_all.tick_params(colors='#000000')
        ax_val_all.xaxis.label.set_color('#000000')
        ax_val_all.yaxis.label.set_color('#000000')
        ax_val_all.title.set_color('#000000')
        ax_val_all.plot(fpr_lgb_val, tpr_lgb_val, label=f"LightGBM (AUC = {auc_lgb_val:.3f})")
        ax_val_all.plot(fpr_xgb_val, tpr_xgb_val, label=f"XGBoost (AUC = {auc_xgb_val:.3f})")
        ax_val_all.plot([0, 1], [0, 1], 'k--')
        ax_val_all.set_xlabel("False Positive Rate")
        ax_val_all.set_ylabel("True Positive Rate")
        ax_val_all.set_title("Validation ROC Curve (LightGBM & XGBoost)")
        ax_val_all.legend(loc="lower right", fontsize='small')


    # Create the DataFrame with calculated AUC values
    df_auc = pd.DataFrame({
        'Model': ['LightGBM', 'XGBoost'],
        'Training AUC': [auc_lgb_train, auc_xgb_train],
        'Validation AUC': [auc_lgb_val, auc_xgb_val]
    })
    for col in ['Training AUC', 'Validation AUC']:
        df_auc[col] = df_auc[col].apply(lambda x: '{:.3f}'.format(x) if x is not None else 'N/A')
    
    # Enhanced table display with status indicators
    st.markdown("""
    <div class='chart-container'>
        <h4 style='margin-bottom: 15px; color: #ff8800;'>📊 Model Performance Metrics</h4>
    """, unsafe_allow_html=True)
    
    # Create styled table with status indicators
    for idx, row in df_auc.iterrows():
        train_auc = float(row['Training AUC']) if row['Training AUC'] != 'N/A' else 0
        val_auc = float(row['Validation AUC']) if row['Validation AUC'] != 'N/A' else 0
        
        # Determine status based on AUC values
        if val_auc >= 0.8:
            status_class = "status-online"
            status_text = "Excellent"
        elif val_auc >= 0.7:
            status_class = "status-warning"
            status_text = "Good"
        else:
            status_class = "status-error"
            status_text = "Needs Improvement"
        
        st.markdown(f"""
        <div style='display: flex; align-items: center; padding: 10px; margin: 5px 0; background: rgba(255,255,255,0.05); border-radius: 8px;'>
            <div class='status-indicator {status_class}'></div>
            <div style='flex: 1;'>
                <strong>{row['Model']}</strong><br>
                <small>Training AUC: {row['Training AUC']} | Validation AUC: {row['Validation AUC']}</small>
            </div>
            <div style='color: #ff8800; font-weight: 600;'>{status_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if fig_train_all is not None:
            st.pyplot(fig_train_all)
    with col2:
        if fig_val_all is not None:
            st.pyplot(fig_val_all)


    # Calculate and show metrics for LightGBM and XGBoost only
    st.subheader("Model Performance Metrics (Validation Data)")
    if len(np.unique(y_val)) < 2:
        st.warning("Validation data contains only one class. Cannot calculate performance metrics.")
    else:
        def calculate_metrics(model, X_val_scaled, y_val):
            preds_proba = model.predict_proba(X_val_scaled)[:, 1]
            preds = (preds_proba > 0.5).astype(int)
            try:
                prec = precision_score(y_val, preds)
            except:
                prec = np.nan
            try:
                rec = recall_score(y_val, preds)
            except:
                rec = np.nan
            try:
                f1 = f1_score(y_val, preds)
            except:
                f1 = np.nan
            acc = accuracy_score(y_val, preds)
            return acc, prec, rec, f1
        acc_lgb, prec_lgb, rec_lgb, f1_lgb = calculate_metrics(lgb_model, X_val_scaled, y_val)
        acc_xgb, prec_xgb, rec_xgb, f1_xgb = calculate_metrics(xgb_model, X_val_scaled, y_val)
        df_calculated_metrics = pd.DataFrame({
            'Model': ['LightGBM', 'XGBoost'],
            'Accuracy': [acc_lgb, acc_xgb],
            'Precision': [prec_lgb, prec_xgb],
            'Recall': [rec_lgb, rec_xgb],
            'F1-score': [f1_lgb, f1_xgb]
        })
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
            df_calculated_metrics[col] = df_calculated_metrics[col].apply(lambda x: '{:.2f}'.format(x) if pd.notna(x) else 'N/A')
        st.table(df_calculated_metrics)


    # ------------------------- USER INPUT PREDICTION (LightGBM & XGBoost) -------------------------
    input_array = np.array(feature_vector).reshape(1, -1)
    input_scaled_for_prediction = scaler.transform(input_array)
    prob_lgb = lgb_model.predict_proba(input_scaled_for_prediction)[:, 1][0]
    prob_xgb = xgb_model.predict_proba(input_scaled_for_prediction)[:, 1][0]

    st.markdown(
        "<small>This model uses synthetic data based on your input structure for demo only. Train on real clinical data for deployment.</small>",
        unsafe_allow_html=True,
    )

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
                df['Predicted_Risk_LightGBM'] = lgb_model.predict_proba(uploaded_features_scaled)[:, 1]

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
                    uploaded_metrics['LightGBM'] = calculate_metrics(lgb_model, uploaded_features_scaled, uploaded_outcomes)

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
                    for model_name, model in zip(['XGBoost', 'LightGBM'], [xgb_model, lgb_model]):
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
    add_column_if_not_exists(cursor, 'logs', 'predicted_risk_lgb', 'REAL')


        # Add opioid calculation before database logging
    opioid = "Yes" if (nalbuphine_dose > 0 or fentanyl_dose > 0 or
                         butorphanol_dose > 0 or pentazocine_dose > 0) else "No"

    # Enhanced logging section with better UI
    st.markdown("""
            <div style='margin: 20px 0; padding: 15px; border-radius: 10px; border-left: 4px solid #ff8800;'>
        <h4 style='margin: 0 0 10px 0; color: #ff8800;'>📝 Data Logging</h4>
        <p style='margin: 0; font-size: 0.9em; color: #ccc;'>Log this assessment for future reference and analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Log This Entry", key='log_entry_button', use_container_width=True):
            with st.spinner("Saving to database..."):
                try:
                    cursor.execute('''
                        INSERT INTO logs (
                            gender, smoker, history_ponv, age, anxiety,
                            abdominal_surgery, volatile, n2o, midazolam,
                            ondansetron, dexamethasone, glycopyrrolate,
                            nalbuphine, fentanyl, butorphanol, pentazocine,
                            propofol_mode, muscle_relaxant, hybrid_score,
                            predicted_risk_xgb, predicted_risk_lgb
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        prob_lgb
                    ))
                    conn.commit()
                    st.markdown("""
                    <div class='success-message'>
                        ✅ <strong>Entry logged successfully!</strong><br>
                        <small>Data saved to local database for future analysis.</small>
                    </div>
                    """, unsafe_allow_html=True)
                except sqlite3.Error as e:
                    st.markdown(f"""
                    <div class='error-message'>
                        ❌ <strong>Database error:</strong> {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
                except NameError as e:
                    st.markdown(f"""
                    <div class='error-message'>
                        ❌ <strong>Logging error:</strong> {str(e)}<br>
                        <small>Please ensure all input fields are selected/filled.</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        if st.button("📊 Show All Entries", key='show_entries_button', use_container_width=True):
            with st.spinner("Loading database entries..."):
                cursor.execute('SELECT * FROM logs ORDER BY timestamp DESC')
                rows = cursor.fetchall()
                if rows:
                    columns = [description[0] for description in cursor.description]
                    df_log = pd.DataFrame(rows, columns=columns)
                    
                    st.markdown("""
                    <div class='success-message'>
                        📋 <strong>Database Entries Loaded</strong><br>
                        <small>Found {} entries in the database.</small>
                    </div>
                    """.format(len(rows)), unsafe_allow_html=True)
                    
                    st.dataframe(df_log, use_container_width=True)

                    if not df_log.empty:
                        csv = df_log.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download All Entries as CSV",
                            data=csv,
                            file_name='logged_ponv_entries.csv',
                            mime='text/csv',
                            key='download_button',
                            use_container_width=True
                        )
                    else:
                        st.warning("No data available to download")
                else:
                    st.markdown("""
                    <div class='error-message'>
                        📭 <strong>No entries found</strong><br>
                        <small>The database is empty. Log some entries first.</small>
                    </div>
                    """, unsafe_allow_html=True)




    # Close the database connection when the app is done (or session ends)
    # This might not be strictly necessary in all Streamlit deployments,
    # but it's good practice.
    # Use an on_after_request or similar if available in the deployment environment
    # For simple Streamlit, the connection might persist across reruns due to session state.
    # Explicitly closing might be tricky without a clear app exit event.
    # conn.close() # Avoid closing here as it will break on rerun

    # Add a note about database persistence
    st.markdown("""
            <div style='font-size: 0.8em; text-align: center; color: #6c757d; margin: 20px 0;'>
            <div class='interactive-card' style='padding: 15px; border-radius: 10px;'>
            <h5 style='margin: 0 0 10px 0; color: #ff8800;'>💾 Data Persistence</h5>
            <p style='margin: 0; font-size: 0.9em;'>Data is logged to a local SQLite file (`ponv_logs.db`). This file will persist as long as the Streamlit application's data directory is maintained.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------- PDF REPORT GENERATION -------------------------
    st.markdown("""
            <div style='margin: 30px 0; padding: 20px; border-radius: 15px; border-left: 5px solid #ff8800;'>
        <h3 style='margin: 0 0 15px 0; color: #fff; text-align: center;'>📄 Generate Comprehensive PDF Report</h3>
        <p style='margin: 0; font-size: 1em; line-height: 1.6; color: #fff; text-align: center;'>
            Download a detailed report containing all patient data, risk assessments, model predictions, graphs, and clinical recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Generate PDF Report", key='generate_pdf_button', use_container_width=True):
            with st.spinner("Generating comprehensive PDF report..."):
                try:
                    # Prepare patient data dictionary
                    patient_data = {
                        'gender': gender,
                        'age': age,
                        'smoker': smoker,
                        'history_ponv': history_ponv,
                        'preop_anxiety': preop_anxiety,
                        'history_migraine': history_migraine,
                        'obesity': obesity,
                        'abdominal_surgery': abdominal_surgery,
                        'ent_surgery': ent_surgery,
                        'gynae_surgery': gynae_surgery,
                        'surgery_duration': surgery_duration,
                        'major_blood_loss': major_blood_loss,
                        'volatile_agents': volatile_agents,
                        'nitrous_oxide': nitrous_oxide,
                        'ondansetron_dose': ondansetron_dose,
                        'midazolam_dose': midazolam_dose,
                        'dexamethasone_dose': dexamethasone_dose,
                        'glycopyrrolate_dose': glycopyrrolate_dose,
                        'nalbuphine_dose': nalbuphine_dose,
                        'fentanyl_dose': fentanyl_dose,
                        'butorphanol_dose': butorphanol_dose,
                        'pentazocine_dose': pentazocine_dose,
                        'propofol_mode': propofol_mode,
                        'muscle_relaxant': muscle_relaxant,
                        'muscle_relaxant_dose': muscle_relaxant_dose
                    }
                    
                    # Prepare model predictions
                    model_predictions = {
                        'lightgbm': prob_lgb,
                        'xgboost': prob_xgb
                    }
                    
                    # Prepare performance metrics
                    performance_metrics = None
                    if len(np.unique(y_val)) >= 2:
                        def calculate_metrics_for_pdf(model, X_val_scaled, y_val):
                            preds_proba = model.predict_proba(X_val_scaled)[:, 1]
                            preds = (preds_proba > 0.5).astype(int)
                            try:
                                prec = precision_score(y_val, preds)
                            except:
                                prec = np.nan
                            try:
                                rec = recall_score(y_val, preds)
                            except:
                                rec = np.nan
                            try:
                                f1 = f1_score(y_val, preds)
                            except:
                                f1 = np.nan
                            acc = accuracy_score(y_val, preds)
                            return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
                        
                        performance_metrics = {
                            'LightGBM': calculate_metrics_for_pdf(lgb_model, X_val_scaled, y_val),
                            'XGBoost': calculate_metrics_for_pdf(xgb_model, X_val_scaled, y_val)
                        }
                    
                    # Prepare feature breakdown
                    feature_breakdown = {
                        "Female Gender": binary(gender),
                        "Non-Smoker": binary(smoker),
                        "History of PONV or Motion Sickness": binary(history_ponv),
                        "Age > 50": 1 if age > 50 else 0,
                        "Preoperative Anxiety": binary(preop_anxiety),
                        "History of Migraine": binary(history_migraine),
                        "BMI > 30": binary(obesity),
                        "Abdominal or Laparoscopic Surgery": binary(abdominal_surgery),
                        "ENT/Neurosurgery/Ophthalmic Surgery": binary(ent_surgery),
                        "Gynecological or Breast Surgery": binary(gynae_surgery),
                        "Surgery Duration > 60 min": binary(surgery_duration),
                        "Major Blood Loss > 500 mL": binary(major_blood_loss),
                        "Use of Volatile Agents": binary(volatile_agents),
                        "Use of Nitrous Oxide": binary(nitrous_oxide),
                        "Midazolam Dose": midazolam_score(midazolam_dose),
                        "Ondansetron Dose": ondansetron_score(ondansetron_dose),
                        "Dexamethasone Dose": dexamethasone_score(dexamethasone_dose),
                        "Glycopyrrolate Dose": glycopyrrolate_score(glycopyrrolate_dose),
                        "Nalbuphine Dose": nalbuphine_score(nalbuphine_dose),
                        "Fentanyl Dose": fentanyl_score(fentanyl_dose),
                        "Butorphanol Dose": butorphanol_score(butorphanol_dose),
                        "Pentazocine Dose": pentazocine_score(pentazocine_dose),
                        "Propofol Use": propofol_score(propofol_mode),
                        "Muscle Relaxant": muscle_relaxant_score(muscle_relaxant, muscle_relaxant_dose),
                    }
                    
                    # Generate feature importance figure for PDF
                    feature_importance_fig = None
                    try:
                        importances = lgb_model.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        top_n = 10
                        top_features = np.array(feature_names)[indices][:top_n][::-1]
                        top_importances = importances[indices][:top_n][::-1]
                        
                        cmap = cm.get_cmap('plasma', top_n)
                        colors_pdf = [cmap(i) for i in range(top_n)]
                        
                        feature_importance_fig, ax = plt.subplots(figsize=(10, 6))
                        feature_importance_fig.patch.set_facecolor('white')
                        ax.set_facecolor('white')
                        bars = ax.barh(top_features, top_importances, color=colors_pdf, edgecolor='black')
                        
                        for bar in bars:
                            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                                    f'{bar.get_width():.0f}', va='center', ha='left', 
                                    color='black', fontsize=10, fontweight='bold')
                        
                        ax.set_yticklabels(top_features, color='black', fontweight='bold', fontname='Arial')
                        ax.set_xlabel('Importance', color='black')
                        ax.set_title('Top 10 Features (LightGBM)', color='black', fontsize=14, fontweight='bold')
                        ax.tick_params(axis='x', colors='black')
                        ax.tick_params(axis='y', colors='black')
                        ax.xaxis.label.set_color('black')
                        ax.yaxis.label.set_color('black')
                        ax.title.set_color('black')
                        ax.spines['bottom'].set_color('black')
                        ax.spines['top'].set_color('black')
                        ax.spines['left'].set_color('black')
                        ax.spines['right'].set_color('black')
                        plt.tight_layout()
                    except Exception as e:
                        st.warning(f"Could not generate feature importance plot for PDF: {e}")
                    
                    # Generate ROC figures for PDF
                    roc_fig_train = None
                    roc_fig_val = None
                    
                    try:
                        if len(np.unique(y_train_balanced)) >= 2:
                            fpr_xgb_train, tpr_xgb_train, _ = roc_curve(y_train_balanced, xgb_model.predict_proba(X_train_balanced)[:, 1])
                            fpr_lgb_train, tpr_lgb_train, _ = roc_curve(y_train_balanced, lgb_model.predict_proba(X_train_balanced)[:, 1])
                            auc_xgb_train = auc(fpr_xgb_train, tpr_xgb_train)
                            auc_lgb_train = auc(fpr_lgb_train, tpr_lgb_train)
                            
                            roc_fig_train, ax_train = plt.subplots(figsize=(8, 6))
                            roc_fig_train.patch.set_facecolor('white')
                            ax_train.set_facecolor('white')
                            ax_train.tick_params(colors='black')
                            ax_train.xaxis.label.set_color('black')
                            ax_train.yaxis.label.set_color('black')
                            ax_train.title.set_color('black')
                            ax_train.plot(fpr_lgb_train, tpr_lgb_train, label=f"LightGBM (AUC = {auc_lgb_train:.3f})")
                            ax_train.plot(fpr_xgb_train, tpr_xgb_train, label=f"XGBoost (AUC = {auc_xgb_train:.3f})")
                            ax_train.plot([0, 1], [0, 1], 'k--')
                            ax_train.set_xlabel("False Positive Rate")
                            ax_train.set_ylabel("True Positive Rate")
                            ax_train.set_title("Training ROC Curve")
                            ax_train.legend(loc="lower right")
                            ax_train.grid(True, alpha=0.3)
                            plt.tight_layout()
                    except Exception as e:
                        st.warning(f"Could not generate training ROC plot for PDF: {e}")
                    
                    try:
                        if len(np.unique(y_val)) >= 2:
                            fpr_xgb_val, tpr_xgb_val, _ = roc_curve(y_val, xgb_model.predict_proba(X_val_scaled)[:, 1])
                            fpr_lgb_val, tpr_lgb_val, _ = roc_curve(y_val, lgb_model.predict_proba(X_val_scaled)[:, 1])
                            auc_xgb_val = auc(fpr_xgb_val, tpr_xgb_val)
                            auc_lgb_val = auc(fpr_lgb_val, tpr_lgb_val)
                            
                            roc_fig_val, ax_val = plt.subplots(figsize=(8, 6))
                            roc_fig_val.patch.set_facecolor('white')
                            ax_val.set_facecolor('white')
                            ax_val.tick_params(colors='black')
                            ax_val.xaxis.label.set_color('black')
                            ax_val.yaxis.label.set_color('black')
                            ax_val.title.set_color('black')
                            ax_val.plot(fpr_lgb_val, tpr_lgb_val, label=f"LightGBM (AUC = {auc_lgb_val:.3f})")
                            ax_val.plot(fpr_xgb_val, tpr_xgb_val, label=f"XGBoost (AUC = {auc_xgb_val:.3f})")
                            ax_val.plot([0, 1], [0, 1], 'k--')
                            ax_val.set_xlabel("False Positive Rate")
                            ax_val.set_ylabel("True Positive Rate")
                            ax_val.set_title("Validation ROC Curve")
                            ax_val.legend(loc="lower right")
                            ax_val.grid(True, alpha=0.3)
                            plt.tight_layout()
                    except Exception as e:
                        st.warning(f"Could not generate validation ROC plot for PDF: {e}")
                    
                    # Generate the PDF
                    pdf_path = generate_pdf_report(
                        patient_data=patient_data,
                        hybrid_score=hybrid_score,
                        risk_category=category,
                        model_predictions=model_predictions,
                        feature_importance_fig=feature_importance_fig,
                        roc_fig_train=roc_fig_train,
                        roc_fig_val=roc_fig_val,
                        performance_metrics=performance_metrics,
                        feature_breakdown=feature_breakdown
                    )
                    
                    # Read the PDF file and create download button
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                    
                    # Create download button
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"PONV_Risk_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key='download_pdf_button',
                        use_container_width=True
                    )
                    
                    st.markdown("""
                    <div class='success-message'>
                        ✅ <strong>PDF Report Generated Successfully!</strong><br>
                        <small>The comprehensive report includes all patient data, risk assessments, model predictions, graphs, and clinical recommendations.</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Clean up temporary PDF file
                    try:
                        os.unlink(pdf_path)
                    except:
                        pass
                        
                except Exception as e:
                    st.markdown(f"""
                    <div class='error-message'>
                        ❌ <strong>PDF Generation Error:</strong> {str(e)}<br>
                        <small>Please ensure all required data is available and try again.</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='padding: 20px; border-radius: 10px; border-left: 4px solid #ff8800;'>
            <h4 style='margin: 0 0 10px 0; color: #ff8800;'>📋 Report Contents</h4>
            <ul style='margin: 0; padding-left: 20px; color: #fff;'>
                <li>Executive Summary</li>
                <li>Patient Information</li>
                <li>Surgical Information</li>
                <li>Drug Administration Details</li>
                <li>Risk Assessment Results</li>
                <li>Feature Importance Analysis</li>
                <li>Model Performance Graphs</li>
                <li>Clinical Recommendations</li>
                <li>Rescue Therapy Guidelines</li>
                <li>Medical Disclaimer & References</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Add floating action button for quick actions
    st.markdown("""
    <div class='fab' onclick='window.scrollTo({top: 0, behavior: "smooth"})' title='Scroll to Top'>
        ↑
    </div>
    """, unsafe_allow_html=True)


    # ------------------------- DISCLAIMER -------------------------
    st.markdown("""
            <div style='margin: 30px 0; padding: 20px; border-radius: 15px; border-left: 5px solid #fff;'>
        <div style='display: flex; align-items: center; margin-bottom: 10px;'>
            <div style='font-size: 1.5em; margin-right: 10px; color: #fff;'>⚠️</div>
            <h4 style='margin: 0; color: #ff3333; font-family: Arial, Helvetica, sans-serif; font-weight: 700;'>Medical Disclaimer</h4>
        </div>
        <p style='margin: 0; font-size: 1em; line-height: 1.6; color: #fff; font-family: Arial, Helvetica, sans-serif;'>
            <strong>Important:</strong> This application is for informational and educational purposes only and should not be considered a substitute for professional medical advice. The predictions and recommendations provided are based on statistical models and should be used as decision support tools only. Always consult with a qualified healthcare provider for diagnosis and treatment decisions.
        </p>
        <div style='margin-top: 10px; font-size: 0.95em; color: #fff; font-family: Arial, Helvetica, sans-serif;'>
            <strong>Developed by:</strong> MKCG Medical College & Hospital - MKCG MedAI Labs<br>
            <strong>Last Updated:</strong> December 2024
        </div>
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
            "Age": 1 if age > 50 else 0,
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
            "Midazolam Dose": midazolam_score(midazolam_dose),
            "Ondansetron Dose": ondansetron_score(ondansetron_dose),
            "Dexamethasone Dose": dexamethasone_score(dexamethasone_dose),
            "Glycopyrrolate Dose": glycopyrrolate_score(glycopyrrolate_dose),
            "Nalbuphine Dose": nalbuphine_score(nalbuphine_dose),
            "Fentanyl Dose": fentanyl_score(fentanyl_dose),
            "Butorphanol Dose": butorphanol_score(butorphanol_dose),
            "Pentazocine Dose": pentazocine_score(pentazocine_dose),
            "Propofol Use": propofol_score(propofol_mode),
            "Muscle Relaxant Used": muscle_relaxant_score(muscle_relaxant, muscle_relaxant_dose),
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
    st.markdown("""
    <div style='font-size:2.2em; font-weight:800; color:#fff; text-align:center; margin-bottom:0.5em;'>Model Training Timeline and Methodological Summary</div>
    """, unsafe_allow_html=True)
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
            </tbody>
        </table>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:1.3em; font-weight:700; color:#fff; margin-top:2em; margin-bottom:0.5em;'>Upcoming Project Phases</div>
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

with tab4:
    st.markdown("""
    <div style='font-size:2.2em; font-weight:800; color:#fff; text-align:center; margin-bottom:0.5em;'>References</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <table style='width:100%; border-collapse: collapse; margin-top: 1.5em; margin-bottom: 1.5em; font-size: 1em; font-family: Arial, Helvetica, sans-serif; box-shadow: 0 0 20px rgba(0,0,0,0.10); border-radius: 10px; overflow: hidden;'>
        <thead>
            <tr style='background-color: #2b5876; color: #fff; text-align: left; font-weight: bold;'>
                <th style='padding: 12px 15px; border: 1px solid #dddddd;'>Score Name</th>
                <th style='padding: 12px 15px; border: 1px solid #dddddd;'>Purpose/Context</th>
                <th style='padding: 12px 15px; border: 1px solid #dddddd;'>Key Reference(s) & Link</th>
            </tr>
        </thead>
        <tbody>
            <tr style='border-bottom: 1px solid #dddddd;'>
                <td style='font-weight: 600; color: #2b5876; padding: 12px 15px; border: 1px solid #dddddd;'>Apfel Score</td>
                <td style='font-size: 0.97em; color: #212529; padding: 12px 15px; border: 1px solid #dddddd;'>PONV risk prediction; 4 risk factors (female gender, non-smoker, history of PONV/motion sickness, postoperative opioids)</td>
                <td style='font-size: 0.95em; color: #495057; padding: 12px 15px; border: 1px solid #dddddd;'>Apfel CC, Läärä E, Koivuranta M, Greim CA, Roewer N. "A simplified risk score for predicting postoperative nausea and vomiting: Conclusions from cross-validations between two centers." Anesthesiology 1999;91:693–700. <a href='https://doi.org/10.1097/00000542-199909000-00022' target='_blank'>[DOI]</a></td>
            </tr>
            <tr style='border-bottom: 1px solid #dddddd;'>
                <td style='font-weight: 600; color: #2b5876; padding: 12px 15px; border: 1px solid #dddddd;'>Koivuranta Score</td>
                <td style='font-size: 0.97em; color: #212529; padding: 12px 15px; border: 1px solid #dddddd;'>PONV risk prediction; 5 predictors (female gender, nonsmoking, history of PONV, history of motion sickness, duration of surgery &gt;60 min)</td>
                <td style='font-size: 0.95em; color: #495057; padding: 12px 15px; border: 1px solid #dddddd;'>Koivuranta M, Läärä E, Snåre L, Alahuhta S. "A survey of postoperative nausea and vomiting." Anaesthesia 1997;52:443–449. <a href='https://doi.org/10.1111/j.1365-2044.1997.00443.x' target='_blank'>[DOI]</a></td>
            </tr>
            <tr style='border-bottom: 1px solid #dddddd;'>
                <td style='font-weight: 600; color: #2b5876; padding: 12px 15px; border: 1px solid #dddddd;'>Sand Score</td>
                <td style='font-size: 0.97em; color: #212529; padding: 12px 15px; border: 1px solid #dddddd;'>PONV risk prediction; simplified Apfel 4-point model</td>
                <td style='font-size: 0.95em; color: #495057; padding: 12px 15px; border: 1px solid #dddddd;'>Same as Apfel Score (see above): <a href='https://doi.org/10.1097/00000542-199909000-00022' target='_blank'>[DOI]</a></td>
            </tr>
            <tr style='border-bottom: 1px solid #dddddd;'>
                <td style='font-weight: 600; color: #2b5876; padding: 12px 15px; border: 1px solid #dddddd;'>Bellville Score</td>
                <td style='font-size: 0.97em; color: #212529; padding: 12px 15px; border: 1px solid #dddddd;'>Severity grading of PONV; measures intensity and frequency</td>
                <td style='font-size: 0.95em; color: #495057; padding: 12px 15px; border: 1px solid #dddddd;'>Kumar A et al. Indian J Anaesth. 2021;65(6):453-459. <a href='https://www.ijaweb.org/article.asp?issn=0019-5049;year=2021;volume=65;issue=6;spage=453;epage=459;aulast=Kumar' target='_blank'>[IJA 2021]</a><br>JCDR 2022;16(1):UC01-UC05. <a href='https://www.jcdr.net/article_fulltext.asp?issn=0973-709x;year=2022;volume=16;issue=1;page=UC01-UC05' target='_blank'>[JCDR 2022]</a><br>Preoperative ondansetron vs dexamethasone: <a href='https://pubmed.ncbi.nlm.nih.gov/23049494/' target='_blank'>[PubMed]</a></td>
            </tr>
        </tbody>
    </table>
    <div style='color:#fff; font-size:0.95em; margin-top:1em;'>
    You can use these links to access the full texts or abstracts of the referenced research papers.
    </div>
    """, unsafe_allow_html=True)

with tab5:
    st.markdown("""
    <div style='padding: 30px 10px 30px 10px; border-radius: 16px; box-shadow: 0 4px 32px rgba(0,0,0,0.25);'>
        <div style='font-size:2.2em; font-weight:800; color:#fff; text-align:center; margin-bottom:0.5em;'>Global Feature Importance (LightGBM)</div>
        <div style='font-size:1.1em; color:#fff; text-align:center; margin-bottom:1em;'>This section shows which variables have the highest global association with the model's predictions, based on LightGBM's feature importances.</div>
    """, unsafe_allow_html=True)
    try:
        import matplotlib.cm as cm
        importances = lgb_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = 10
        top_features = np.array(feature_names)[indices][:top_n][::-1]
        top_importances = importances[indices][:top_n][::-1]

        # Use a colorful colormap
        cmap = cm.get_cmap('plasma', top_n)
        colors = [cmap(i) for i in range(top_n)]

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#000')
        ax.set_facecolor('#000')
        bars = ax.barh(top_features, top_importances, color=colors, edgecolor='white')

        # Add value labels
        for bar in bars:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{bar.get_width():.0f}', va='center', ha='left', color='white', fontsize=11, fontweight='bold')

        # Set y-tick labels (feature names) to black, bold, simple font
        ax.set_yticklabels(top_features, color='black', fontweight='bold', fontname='Arial')

        ax.set_xlabel('Importance', color='white')
        ax.set_title('Top 10 Features (LightGBM)', color='#ffb366', fontsize=16, fontweight='bold')
        ax.tick_params(axis='x', colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('#ffb366')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        plt.tight_layout()
        st.pyplot(fig)
        # Table of top 10 features
        import pandas as pd
        df_feat = pd.DataFrame({
            'Feature': np.array(feature_names)[indices][:top_n],
            'Importance': importances[indices][:top_n]
        })
        st.subheader("Top 10 Most Important Features")
        st.table(df_feat)
    except Exception as e:
        st.warning(f"Could not display feature importance: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#000000",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "text.color": "#000000",
    "axes.titlecolor": "#000000",
    "legend.edgecolor": "#000000",
    "legend.facecolor": "#ffffff",
    "legend.labelcolor": "#000000",
    "savefig.facecolor": "#ffffff",
    "savefig.edgecolor": "#ffffff",
    "grid.color": "#cccccc"
})


