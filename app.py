import os
import re
import numpy as np
import pandas as pd
import cv2
import pytesseract
from PIL import Image
import hashlib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import streamlit as st
import io
import base64
from pdf2image import convert_from_bytes
import sqlite3
import json
import traceback
import fitz

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

def init_db():
    os.makedirs('database', exist_ok=True)
    conn = sqlite3.connect('database/document_verification.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS verified_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prn TEXT,
        student_name TEXT, 
        college_name TEXT,
        verification_date TEXT,
        verification_status TEXT,
        confidence_score REAL,
        document_hash TEXT,
        metadata TEXT
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS verification_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        document_id INTEGER,
        action TEXT,
        details TEXT,
        FOREIGN KEY (document_id) REFERENCES verified_documents (id)
    )
    ''')
    conn.commit()
    conn.close()

class DocumentProcessor:
    def __init__(self):
        self.extraction_confidence = 0
        self.verification_score = 0
        self.anomaly_score = 0
        self.validation_results = {}
        
    def preprocess_image(self, image):
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                else:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            final = clahe.apply(opening)
            return final
        except Exception as e:
            st.error(f"Image preprocessing error: {str(e)}")
            return image

    def pdf_to_image(self, pdf_bytes):
        try:
            images = convert_from_bytes(pdf_bytes)
            return np.array(images[0])
        except Exception:
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                return np.array(Image.open(io.BytesIO(img_bytes)))
            except Exception as e:
                st.error(f"PDF conversion failed: {str(e)}")
                raise

    def extract_text(self, image):
        try:
            preprocessed = self.preprocess_image(image)
            custom_config = r'--oem 3 --psm 6 -l eng'
            text = pytesseract.image_to_string(preprocessed, config=custom_config)
            data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT, config=custom_config)
            confidences = [float(conf) for conf in data['conf'] if conf != '-1']
            self.extraction_confidence = np.mean(confidences) if confidences else 0
            return text
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")
            if "TesseractNotFoundError" in str(e):
                st.error("Tesseract OCR is not installed or not in your PATH")
            return ""

    def extract_fields(self, text):
        fields = {
            'prn': None,
            'student_name': None,
            'mother_name': None,
            'college_name': None,
            'branch': None,
            'subjects': [],
            'grades': [],
            'credits': [],
            'sgpa': None,
            'result_date': None
        }
        prn_match = re.search(r'(?:Perm\s*Reg\s*No\(PRN\)|PRN|Seat\s*No)[:\s]*([0-9A-Za-z]+)', text, re.IGNORECASE)
        if prn_match:
            fields['prn'] = prn_match.group(1).strip()
        name_match = re.search(r'(?:Student\s*Name|Name\s*of\s*Student)[:\s]*([^\n]+)', text, re.IGNORECASE)
        if name_match:
            fields['student_name'] = name_match.group(1).strip()
        college_match = re.search(r'College\s*Name[:\s]*(\d+\s+[^\n]+)', text, re.IGNORECASE)
        if college_match:
            fields['college_name'] = college_match.group(1).strip()
        branch_match = re.search(r'Branch/Course[:\s]*([^\n]+)', text, re.IGNORECASE)
        if branch_match:
            fields['branch'] = branch_match.group(1).strip()
        sgpa_match = re.search(r'SGPA\s*\d*[:\s-]*([0-9.]+)', text, re.IGNORECASE)
        if not sgpa_match:
            sgpa_match = re.search(r'SGPA1[:\s-]*([0-9.]+)', text, re.IGNORECASE)
        if sgpa_match:
            try:
                fields['sgpa'] = float(sgpa_match.group(1))
            except ValueError:
                pass
        date_match = re.search(r'RESULT\s*DATE[:\s]*([0-9]+\s+[A-Za-z]+\s+[0-9]+)', text, re.IGNORECASE)
        if date_match:
            fields['result_date'] = date_match.group(1).strip()
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        subject_pattern = r'(\d{6})\s+([A-Z][A-Z\s.&]+)\s+(\d)\s+([A-Z+]+)\s+(\d+)'
        subject_matches = re.finditer(subject_pattern, text)
        for match in subject_matches:
            subj_name = match.group(2).replace('.', ' ')
            fields['subjects'].append({
                'code': match.group(1),
                'name': subj_name.strip(),
                'credits': int(match.group(3)),
                'grade': match.group(4),
                'grade_points': int(match.group(5))
            })
        print(f"Extracted SGPA: {fields.get('sgpa')}")
        print(f"Extracted Student Name: {fields.get('student_name')}")
        print(f"Extracted Mother Name: {fields.get('mother_name')}")
        print(f"Calculated Credits: {sum(s['credits'] for s in fields['subjects'])}")
        print(f"Subjects: {[(s['code'], s['credits']) for s in fields['subjects']]}")
        return fields

    def verify_document(self, fields, text=""):
        verification_checks = {
            'prn_format': False,
            'name_present': False,
            'college_present': False,
            'subjects_present': False,
            'grade_validation': False,
            'credit_sum': False,
            'sgpa_calculation': False,
            'date_format': False
        }
        if fields['prn'] and re.match(r'^[0-9A-Za-z]{8,12}$', fields['prn']):
            verification_checks['prn_format'] = True
        verification_checks['name_present'] = bool(fields.get('student_name')) or bool(fields.get('mother_name'))
        verification_checks['college_present'] = bool(fields['college_name'])
        verification_checks['subjects_present'] = len(fields['subjects']) >= 3
        valid_grades = {'O', 'A+', 'A', 'B+', 'B', 'C', 'D', 'F', 'P'}
        if fields['subjects']:
            all_valid = all(subject['grade'] in valid_grades for subject in fields['subjects'])
            verification_checks['grade_validation'] = all_valid
            total_credits_calculated = sum(subject['credits'] for subject in fields['subjects'])
            total_credits_stated = None
            if text:
                credit_match = re.search(r'TOTALCREDITS\s+EARNED[:\s-]*(\d+)', text, re.IGNORECASE)
                if credit_match:
                    try:
                        total_credits_stated = int(credit_match.group(1))
                    except (ValueError, AttributeError):
                        pass
            if total_credits_stated is not None:
                verification_checks['credit_sum'] = abs(total_credits_calculated - total_credits_stated) <= 1
            else:
                verification_checks['credit_sum'] = (15 <= total_credits_calculated <= 30)
        else:
            verification_checks['credit_sum'] = False
        if fields['subjects'] and fields['sgpa']:
            total_grade_points = sum(subject['grade_points'] for subject in fields['subjects'])
            total_credits = sum(subject['credits'] for subject in fields['subjects'])
            if total_credits > 0:
                calculated_sgpa = round(total_grade_points / total_credits, 2)
                verification_checks['sgpa_calculation'] = (abs(calculated_sgpa - fields['sgpa']) < 0.3)
        if fields['result_date']:
            date_pattern = r'([0-9]+\s+[A-Za-z]+\s+[0-9]+|[0-9]{2}/[0-9]{2}/[0-9]{4})'
            verification_checks['date_format'] = bool(re.match(date_pattern, fields['result_date']))
        self.validation_results = verification_checks
        weights = {
            'prn_format': 0.15,
            'name_present': 0.1,
            'college_present': 0.1,
            'subjects_present': 0.15,
            'grade_validation': 0.2,
            'credit_sum': 0.1,
            'sgpa_calculation': 0.15,
            'date_format': 0.05
        }
        passed_score = sum(weights[check] * (1 if passed else 0) 
                        for check, passed in verification_checks.items())
        self.verification_score = passed_score
        return verification_checks
    def detect_anomalies(self, fields):
        if not fields['subjects']:
            self.anomaly_score = 0.5
            return 0.5
        try:
            grade_values = {'O': 10, 'A+': 9, 'A': 8, 'B+': 7, 'B': 6, 
                            'C': 5, 'D': 4, 'F': 0, 'P': 0}
            subject_data = []
            for subject in fields['subjects']:
                subject_data.append([
                    subject['credits'],
                    grade_values.get(subject['grade'], 0),
                    subject['grade_points']
                ])
            if len(subject_data) < 3:
                self.anomaly_score = 0.5
                return 0.5
            X = np.array(subject_data)
            clf = IsolationForest(random_state=42, contamination=0.1)
            clf.fit(X)
            anomaly_predictions = clf.predict(X)
            if_score = 1 - (np.sum(anomaly_predictions == -1) / len(anomaly_predictions))
            self.anomaly_score = if_score
            return if_score
        except Exception as e:
            st.error(f"Anomaly detection error: {str(e)}")
            return 0.5
    
    def calculate_final_confidence(self):
        extraction_weight = 0.2 * (self.extraction_confidence / 100)
        verification_weight = 0.7 * self.verification_score
        anomaly_weight = 0.1 * self.anomaly_score
        confidence = extraction_weight + verification_weight + anomaly_weight
        critical_checks = ['prn_format', 'name_present', 'college_present', 'subjects_present']
        if all(self.validation_results.get(check, False) for check in critical_checks):
            confidence = min(confidence * 1.1, 1.0)
        return min(max(confidence, 0), 1)
    
    def generate_document_hash(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return hashlib.sha256(image.tobytes()).hexdigest()

def main():
    st.set_page_config(
        page_title="Academic Document Verification",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    init_db()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Verify Document", "History", "Analytics", "About"],
        index=0
    )
    if page == "Verify Document":
        show_verification_page()
    elif page == "History":
        show_history_page()
    elif page == "Analytics":
        show_analytics_page()
    else:
        show_about_page()

if __name__ == "__main__":
    main()
