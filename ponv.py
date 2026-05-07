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
    This application is for informational and educational purposes only and should not be considered a substitute for professional medical advice. The predictions and recommendations provided are base[...]
    
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

# CONTINUE WITH REST OF THE FILE (keeping all content same until tab2)
# ... [All the CSS and initial setup code remains the same - lines 445-2688] ...

# I'll continue with the corrected tab2 section where the fix is needed
