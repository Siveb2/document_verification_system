# Academic Document Verification System
# Main application file: app.py

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
import fitz  # PyMuPDF

# Configure Tesseract path (update this for your system)
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:  # Linux/Mac
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

# Initialize database
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

# Document processor class
class DocumentProcessor:
    def __init__(self):
        self.extraction_confidence = 0
        self.verification_score = 0
        self.anomaly_score = 0
        self.validation_results = {}
        
    def preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA image
                    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                else:  # RGB image
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image  # Already grayscale

            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
            
            # Remove noise
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            final = clahe.apply(opening)
            
            return final
        except Exception as e:
            st.error(f"Image preprocessing error: {str(e)}")
            return image

    def pdf_to_image(self, pdf_bytes):
        """Convert PDF bytes to image using multiple methods"""
        try:
            # Try pdf2image first
            images = convert_from_bytes(pdf_bytes)
            return np.array(images[0])
        except Exception:
            try:
                # Fallback to PyMuPDF
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                return np.array(Image.open(io.BytesIO(img_bytes)))
            except Exception as e:
                st.error(f"PDF conversion failed: {str(e)}")
                raise

    def extract_text(self, image):
        """Extract text from image using OCR with improved configuration"""
        try:
            preprocessed = self.preprocess_image(image)
            
            # Custom OCR configuration
            custom_config = r'--oem 3 --psm 6 -l eng'
            text = pytesseract.image_to_string(
                preprocessed, 
                config=custom_config
            )
            
            # Calculate confidence
            data = pytesseract.image_to_data(
                preprocessed, 
                output_type=pytesseract.Output.DICT,
                config=custom_config
            )
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
        
        # Extract PRN (more flexible pattern)
        prn_match = re.search(r'(?:Perm\s*Reg\s*No\(PRN\)|PRN|Seat\s*No)[:\s]*([0-9A-Za-z]+)', text, re.IGNORECASE)
        if prn_match:
            fields['prn'] = prn_match.group(1).strip()
        
        # Extract Student Name (handle multiline names)
        name_match = re.search(r'(?:Student\s*Name|Name\s*of\s*Student)[:\s]*([^\n]+)', text, re.IGNORECASE)
        if name_match:
            fields['student_name'] = name_match.group(1).strip()
        
        # Extract College Name (handle your college format)
        college_match = re.search(r'College\s*Name[:\s]*(\d+\s+[^\n]+)', text, re.IGNORECASE)
        if college_match:
            fields['college_name'] = college_match.group(1).strip()
        
        # Extract Branch/Course
        branch_match = re.search(r'Branch/Course[:\s]*([^\n]+)', text, re.IGNORECASE)
        if branch_match:
            fields['branch'] = branch_match.group(1).strip()
        
        # Extract SGPA (handle your format)
        
        # Replace the existing sgpa_match code with this:
        sgpa_match = re.search(r'SGPA\s*\d*[:\s-]*([0-9.]+)', text, re.IGNORECASE)
        if not sgpa_match:  # Alternative pattern for your specific format
            sgpa_match = re.search(r'SGPA1[:\s-]*([0-9.]+)', text, re.IGNORECASE)
        if sgpa_match:
            try:
                fields['sgpa'] = float(sgpa_match.group(1))
            except ValueError:
                pass
        
        # Extract Result Date (handle your format)
        date_match = re.search(r'RESULT\s*DATE[:\s]*([0-9]+\s+[A-Za-z]+\s+[0-9]+)', text, re.IGNORECASE)
        if date_match:
            fields['result_date'] = date_match.group(1).strip()
         # Clean up text for better subject extraction
        text = text.replace('\n', ' ')  # Handle line breaks in subject names
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        # Improved subject extraction pattern for your format
        subject_pattern = r'(\d{6})\s+([A-Z][A-Z\s.&]+)\s+(\d)\s+([A-Z+]+)\s+(\d+)'
        subject_matches = re.finditer(subject_pattern, text)
               
        for match in subject_matches:
            subj_name = match.group(2).replace('.', ' ')  # Clean up subject names
            fields['subjects'].append({
                'code': match.group(1),
                'name': subj_name.strip(),
                'credits': int(match.group(3)),
                'grade': match.group(4),
                'grade_points': int(match.group(5))
            })
        # Temporary debug prints (can remove after testing)
        print(f"Extracted SGPA: {fields.get('sgpa')}")  # Should show 8.10
        print(f"Extracted Student Name: {fields.get('student_name')}")  # Should show BARAIYAVIVEK JAGDISH
        print(f"Extracted Mother Name: {fields.get('mother_name')}")  # Should show GEETA
        # Add this temporary debug right before the return in extract_fields:
        print(f"Calculated Credits: {sum(s['credits'] for s in fields['subjects'])}")
        print(f"Subjects: {[(s['code'], s['credits']) for s in fields['subjects']]}")

        return fields
    # def extract_fields(self, text):
    #     """Extract relevant fields from OCR text with improved patterns"""
    #     fields = {
    #         'prn': None,
    #         'student_name': None,
    #         'mother_name': None,
    #         'college_name': None,
    #         'branch': None,
    #         'subjects': [],
    #         'grades': [],
    #         'credits': [],
    #         'sgpa': None,
    #         'result_date': None
    #     }
        
    #     # Extract PRN (more flexible pattern)
    #     prn_match = re.search(r'(?:Perm\s*Reg\s*No|PRN)[:\s]*([0-9A-Za-z]{10,20})', text, re.IGNORECASE)
    #     if prn_match:
    #         fields['prn'] = prn_match.group(1).strip()
        
    #     # Extract Student Name (more flexible pattern)
    #     name_match = re.search(r'(?:Student\s*Name|Name\s*of\s*Student)[:\s]*([^\n]+)', text, re.IGNORECASE)
    #     if name_match:
    #         fields['student_name'] = name_match.group(1).strip()
        
    #     # Extract College Name
    #     college_match = re.search(r'(?:College\s*Name|Name\s*of\s*College)[:\s]*([^\n]+)', text, re.IGNORECASE)
    #     if college_match:
    #         fields['college_name'] = college_match.group(1).strip()
        
    #     # Extract Branch/Course
    #     branch_match = re.search(r'(?:Branch|Course)[:\s]*([^\n]+)', text, re.IGNORECASE)
    #     if branch_match:
    #         fields['branch'] = branch_match.group(1).strip()
        
    #     # Extract SGPA/CGPA
    #     sgpa_match = re.search(r'(SGPA|CGPA)[\s:]*([0-9.]+)', text, re.IGNORECASE)
    #     if sgpa_match:
    #         fields['sgpa'] = float(sgpa_match.group(2))
        
    #     # Extract Result Date
    #     date_match = re.search(r'(?:Result\s*Date|Date\s*of\s*Result)[:\s]*([0-9]{2}/[0-9]{2}/[0-9]{4}|[0-9]+\s+[A-Za-z]+\s+[0-9]+)', text, re.IGNORECASE)
    #     if date_match:
    #         fields['result_date'] = date_match.group(1).strip()
        
    #     # Improved subject extraction pattern
    #     subject_pattern = r'([0-9]{6})\s+([A-Z][A-Z\s.&]+)\s+([0-9])\s+([A-Z+]+)\s+([0-9]+)'
    #     subject_matches = re.finditer(subject_pattern, text)
        
    #     for match in subject_matches:
    #         fields['subjects'].append({
    #             'code': match.group(1),
    #             'name': match.group(2).strip(),
    #             'credits': int(match.group(3)),
    #             'grade': match.group(4),
    #             'grade_points': int(match.group(5))
    #         })
        
    #     return fields
    
    def verify_document(self, fields, text=""):
        verification_checks = {'prn_format': False,'name_present': False,'college_present': False,'subjects_present': False,'grade_validation': False,'credit_sum': False,'sgpa_calculation': False,'date_format': False}
    
        # PRN format check - accept alphanumeric with length 8-12
        if fields['prn'] and re.match(r'^[0-9A-Za-z]{8,12}$', fields['prn']):
            verification_checks['prn_format'] = True
        
        # Name presence check
        # Replace the name validation check with this:
        verification_checks['name_present'] = bool(fields.get('student_name')) or bool(fields.get('mother_name'))
        
        # College presence check
        verification_checks['college_present'] = bool(fields['college_name'])
        
        # Subjects presence check
        verification_checks['subjects_present'] = len(fields['subjects']) >= 3  # At least 3 subjects
        
        # Grade validation
        valid_grades = {'O', 'A+', 'A', 'B+', 'B', 'C', 'D', 'F', 'P'}
        if fields['subjects']:
            all_valid = all(subject['grade'] in valid_grades for subject in fields['subjects'])
            verification_checks['grade_validation'] = all_valid
        
        # Credit sum check (more flexible range)
        total_credits_calculated = 0  # Initialize outside the if block
        if fields.get('subjects'):
            total_credits_calculated = sum(subject['credits'] for subject in fields['subjects'])
        
        # Try to get from TOTALCREDITS text if available
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
            verification_checks['credit_sum'] = False  # No subjects means automatic fail
        
        # SGPA calculation check (more tolerant threshold)
        # Update the SGPA calculation check to:
        if fields['subjects'] and fields['sgpa']:
            total_grade_points = sum(subject['grade_points'] for subject in fields['subjects'])
            total_credits = sum(subject['credits'] for subject in fields['subjects'])
            if total_credits > 0:
                calculated_sgpa = round(total_grade_points / total_credits, 2)
                verification_checks['sgpa_calculation'] = (abs(calculated_sgpa - fields['sgpa']) < 0.3)  # Increased tolerance
        
        # Date format check (more flexible)
        if fields['result_date']:
            date_pattern = r'([0-9]+\s+[A-Za-z]+\s+[0-9]+|[0-9]{2}/[0-9]{2}/[0-9]{4})'
            verification_checks['date_format'] = bool(re.match(date_pattern, fields['result_date']))
        
        self.validation_results = verification_checks
        
        # Calculate verification score (weighted)
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
        """Enhanced anomaly detection with multiple techniques"""
        if not fields['subjects']:
            self.anomaly_score = 0.5
            return 0.5
        
        try:
            # Prepare data
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
            
            # Isolation Forest
            clf = IsolationForest(random_state=42, contamination=0.1)
            clf.fit(X)
            anomaly_predictions = clf.predict(X)
            if_score = 1 - (np.sum(anomaly_predictions == -1) / len(anomaly_predictions))
            
            # Combine scores
            self.anomaly_score = if_score
            return if_score
            
        except Exception as e:
            st.error(f"Anomaly detection error: {str(e)}")
            return 0.5
    
    def calculate_final_confidence(self):
        """Calculate final confidence score with improved weighting"""
        extraction_weight = 0.2 * (self.extraction_confidence / 100)
        verification_weight = 0.7 * self.verification_score
        anomaly_weight = 0.1 * self.anomaly_score
        
        confidence = extraction_weight + verification_weight + anomaly_weight
        critical_checks = ['prn_format', 'name_present', 'college_present', 'subjects_present']
        if all(self.validation_results.get(check, False) for check in critical_checks):
            confidence = min(confidence * 1.1, 1.0)
        return min(max(confidence, 0), 1)
    
    def generate_document_hash(self, image):
        """Generate SHA-256 hash of document image"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return hashlib.sha256(image.tobytes()).hexdigest()

# Database operations
def save_verification_result(fields, confidence, document_hash, verification_status):
    conn = sqlite3.connect('database/document_verification.db')
    c = conn.cursor()
    
    # Check for existing document
    c.execute("SELECT id FROM verified_documents WHERE document_hash = ?", (document_hash,))
    result = c.fetchone()
    
    metadata = json.dumps({
        'subjects': fields.get('subjects', []),
        'sgpa': fields.get('sgpa'),
        'branch': fields.get('branch'),
        'result_date': fields.get('result_date'),
        'validation_results': {
            check: status 
            for check, status in fields.get('validation_results', {}).items()
        }
    })
    
    if result:
        doc_id = result[0]
        c.execute("""
        UPDATE verified_documents SET 
            verification_date = datetime('now'), 
            verification_status = ?, 
            confidence_score = ?,
            metadata = ?
        WHERE id = ?
        """, (verification_status, confidence, metadata, doc_id))
    else:
        c.execute("""
        INSERT INTO verified_documents 
        (prn, student_name, college_name, verification_date, 
         verification_status, confidence_score, document_hash, metadata)
        VALUES (?, ?, ?, datetime('now'), ?, ?, ?, ?)
        """, (
            fields.get('prn', ''),
            fields.get('student_name', ''),
            fields.get('college_name', ''),
            verification_status,
            confidence,
            document_hash,
            metadata
        ))
        doc_id = c.lastrowid
    
    # Log verification
    c.execute("""
    INSERT INTO verification_logs
    (timestamp, document_id, action, details)
    VALUES (datetime('now'), ?, ?, ?)
    """, (
        doc_id,
        "VERIFICATION",
        f"Status: {verification_status}, Confidence: {confidence:.2f}"
    ))
    
    conn.commit()
    conn.close()
    return doc_id

def get_verification_history(limit=50):
    conn = sqlite3.connect('database/document_verification.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("""
    SELECT id, prn, student_name, college_name, 
           strftime('%Y-%m-%d %H:%M', verification_date) as verification_date,
           verification_status, confidence_score
    FROM verified_documents
    ORDER BY verification_date DESC
    LIMIT ?
    """, (limit,))
    
    results = [dict(row) for row in c.fetchall()]
    conn.close()
    return results

def get_document_details(doc_id):
    conn = sqlite3.connect('database/document_verification.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("""
    SELECT *, strftime('%Y-%m-%d %H:%M', verification_date) as formatted_date
    FROM verified_documents
    WHERE id = ?
    """, (doc_id,))
    
    result = c.fetchone()
    if result:
        doc = dict(result)
        doc['metadata'] = json.loads(doc['metadata'])
        return doc
    
    conn.close()
    return None

# Streamlit UI Components
def show_verification_page():
    st.title("📄 Academic Document Verification")
    st.write("Upload a marksheet or academic transcript for verification")
    
    with st.expander("ℹ️ Upload Instructions"):
        st.markdown("""
        - Supported formats: PDF, PNG, JPG
        - Ensure document is clear and all text is visible
        - For best results, use high-quality scans
        - The system works best with standard university marksheets
        """)
    
    uploaded_file = st.file_uploader(
        "Choose a document", 
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=False
    )
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            try:
                processor = DocumentProcessor()
                file_bytes = uploaded_file.read()
                
                # Convert to image
                if uploaded_file.type == "application/pdf":
                    image = processor.pdf_to_image(file_bytes)
                else:
                    image = np.array(Image.open(io.BytesIO(file_bytes)))
                
                # Display document preview
                st.subheader("Document Preview")
                st.image(image, use_column_width=True)
                
                # Process document
                text = processor.extract_text(image)
                
                if not text or len(text.strip()) < 50:
                    st.error("Text extraction failed. The document may be unclear or in an unsupported format.")
                    return
                
                fields = processor.extract_fields(text)
                verification = processor.verify_document(fields, text)  # Pass the OCR text
                anomaly_score = processor.detect_anomalies(fields)
                confidence = processor.calculate_final_confidence()
                doc_hash = processor.generate_document_hash(image)
                
                # Determine status
                if confidence >= 0.85:
                    status = "✅ VERIFIED"
                    status_color = "green"
                elif confidence >= 0.6:
                    status = "⚠️ NEEDS REVIEW"
                    status_color = "orange"
                else:
                    status = "❌ POTENTIAL FRAUD"
                    status_color = "red"
                
                # Save results
                save_verification_result(fields, confidence, doc_hash, status)
                
                # Display results
                st.subheader("Verification Results")
                
                # Create columns layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Document Information")
                    info_data = {
                        "Student Name": fields.get('student_name', 'Not found'),
                        "PRN": fields.get('prn', 'Not found'),
                        "College": fields.get('college_name', 'Not found'),
                        "Branch": fields.get('branch', 'Not found'),
                        "SGPA": fields.get('sgpa', 'Not found'),
                        "Result Date": fields.get('result_date', 'Not found')
                    }
                    st.table(pd.DataFrame.from_dict(info_data, orient='index', columns=['Value']))
                
                with col2:
                    st.markdown("### Verification Status")
                    st.markdown(f"<h3 style='color:{status_color};'>{status}</h3>", 
                               unsafe_allow_html=True)
                    
                    # Confidence meter
                    st.markdown(f"**Confidence Score:** {confidence:.2%}")
                    st.progress(confidence)
                    
                    # Anomaly score
                    st.markdown(f"**Anomaly Detection:** {anomaly_score:.2%}")
                    st.progress(anomaly_score)
                    
                    # Document fingerprint
                    st.markdown("**Document Hash:**")
                    st.code(doc_hash[:20] + "..." + doc_hash[-20:])
                
                # Subjects table
                if fields['subjects']:
                    st.subheader("Subjects")
                    subjects_df = pd.DataFrame([
                        {
                            "Code": s["code"],
                            "Subject": s["name"],
                            "Credits": s["credits"],
                            "Grade": s["grade"],
                            "Points": s["grade_points"]
                        }
                        for s in fields['subjects']
                    ])
                    st.dataframe(subjects_df)
                
                # Verification checks
                st.subheader("Verification Checks")
                checks_df = pd.DataFrame([
                    {"Check": check.replace('_', ' ').title(), 
                     "Status": "✅ Passed" if passed else "❌ Failed"}
                    for check, passed in verification.items()
                ])
                st.table(checks_df)
                
                # Technical details
                with st.expander("Technical Details"):
                    st.markdown(f"**OCR Confidence:** {processor.extraction_confidence:.2f}%")
                    st.markdown("**Extracted Text (Sample):**")
                    st.text(text[:500] + "..." if len(text) > 500 else text)
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.error(traceback.format_exc())

def show_history_page():
    st.title("📜 Verification History")
    
    history = get_verification_history(100)
    
    if not history:
        st.info("No verification history found.")
        return
    
    # Search and filter
    search_col, filter_col = st.columns(2)
    
    with search_col:
        search_term = st.text_input("Search by PRN or Name")
    
    with filter_col:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "VERIFIED", "NEEDS REVIEW", "POTENTIAL FRAUD"]
        )
    
    # Apply filters
    filtered_history = history
    if search_term:
        filtered_history = [
            h for h in filtered_history 
            if search_term.lower() in h['prn'].lower() or 
               search_term.lower() in h['student_name'].lower()
        ]
    
    if status_filter != "All":
        filtered_history = [
            h for h in filtered_history 
            if status_filter in h['verification_status']
        ]
    
    if not filtered_history:
        st.info("No documents match your filters.")
        return
    
    # Display history table
    history_df = pd.DataFrame([
        {
            "ID": h["id"],
            "PRN": h["prn"],
            "Name": h["student_name"],
            "College": h["college_name"],
            "Date": h["verification_date"],
            "Status": h["verification_status"],
            "Confidence": f"{h['confidence_score']:.2%}"
        }
        for h in filtered_history
    ])
    
    st.dataframe(
        history_df,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                format="%.2f",
                min_value=0,
                max_value=1
            )
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Document details viewer
    st.subheader("Document Details")
    selected_id = st.selectbox(
        "Select a document to view details",
        options=[h["id"] for h in filtered_history],
        format_func=lambda x: f"ID {x} - {next(h['student_name'] for h in filtered_history if h['id'] == x)}"
    )
    
    if selected_id:
        doc = get_document_details(selected_id)
        if doc:
            cols = st.columns(2)
            
            with cols[0]:
                st.markdown("### Basic Information")
                st.table(pd.DataFrame.from_dict({
                    "PRN": doc['prn'],
                    "Student Name": doc['student_name'],
                    "College": doc['college_name'],
                    "Verification Date": doc['formatted_date'],
                    "Status": doc['verification_status'],
                    "Confidence": f"{doc['confidence_score']:.2%}"
                }, orient='index'))
            
            with cols[1]:
                st.markdown("### Academic Details")
                if 'sgpa' in doc['metadata']:
                    st.metric("SGPA", doc['metadata']['sgpa'])
                if 'branch' in doc['metadata']:
                    st.metric("Branch", doc['metadata']['branch'])
                if 'result_date' in doc['metadata']:
                    st.metric("Result Date", doc['metadata']['result_date'])
            
            # Subjects table if available
            if 'subjects' in doc['metadata'] and doc['metadata']['subjects']:
                st.markdown("### Subjects")
                subjects_df = pd.DataFrame(doc['metadata']['subjects'])
                st.dataframe(subjects_df[['code', 'name', 'credits', 'grade', 'grade_points']])
            
            # Validation results
            if 'validation_results' in doc['metadata']:
                st.markdown("### Validation Checks")
                checks_df = pd.DataFrame([
                    {"Check": check.replace('_', ' ').title(), 
                     "Status": "✅ Passed" if passed else "❌ Failed"}
                    for check, passed in doc['metadata']['validation_results'].items()
                ])
                st.table(checks_df)

def show_analytics_page():
    st.title("📊 Verification Analytics")
    
    history = get_verification_history(500)
    if not history:
        st.info("No verification data available for analytics.")
        return
    
    df = pd.DataFrame(history)
    df['verification_date'] = pd.to_datetime(df['verification_date'])
    df['date'] = df['verification_date'].dt.date
    
    # Overall stats
    st.subheader("Overall Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Verifications", len(df))
    
    with col2:
        verified_pct = len(df[df['verification_status'] == "✅ VERIFIED"]) / len(df)
        st.metric("Verified Percentage", f"{verified_pct:.1%}")
    
    with col3:
        avg_confidence = df['confidence_score'].mean()
        st.metric("Average Confidence", f"{avg_confidence:.1%}")
    
    # Charts
    st.subheader("Verification Trends")
    
    tab1, tab2, tab3 = st.tabs(["Status Distribution", "Daily Activity", "College Distribution"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(8, 6))
        status_counts = df['verification_status'].value_counts()
        colors = ['green', 'orange', 'red']
        status_counts.plot.pie(autopct='%1.1f%%', colors=colors, ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)
    
    with tab2:
        daily = df.groupby('date').size().reset_index(name='count')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=daily, x='date', y='count', marker='o', ax=ax)
        ax.set_title("Daily Verifications")
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    
    with tab3:
        top_colleges = df['college_name'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_colleges.values, y=top_colleges.index, ax=ax)
        ax.set_title("Top 10 Colleges")
        ax.set_xlabel("Count")
        st.pyplot(fig)
    
    # Confidence analysis
    st.subheader("Confidence Analysis")
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.histplot(df['confidence_score'], bins=20, kde=True, ax=ax[0])
    ax[0].set_title("Confidence Score Distribution")
    
    sns.boxplot(x='verification_status', y='confidence_score', data=df, ax=ax[1])
    ax[1].set_title("Confidence by Verification Status")
    
    st.pyplot(fig)

def show_about_page():
    st.title("ℹ️ About the Document Verification System")
    
    st.markdown("""
    ## Project Overview
    
    This Academic Document Verification System uses advanced techniques to verify the authenticity 
    of academic documents like marksheets and transcripts.
    
    ### Key Features:
    
    - **Document Analysis**: Uses OCR to extract text and information
    - **Data Validation**: Performs multiple validation checks
    - **Anomaly Detection**: Uses machine learning to identify potential fraud
    - **Verification History**: Maintains a searchable database of documents
    - **Analytics Dashboard**: Provides insights into verification trends
    
    ### Technical Stack:
    
    - **OCR**: Tesseract with custom preprocessing
    - **Machine Learning**: Isolation Forest for anomaly detection
    - **Database**: SQLite for document storage
    - **UI**: Streamlit for interactive web interface
    
    ### System Requirements:
    
    - Tesseract OCR installed
    - Poppler utilities for PDF processing
    - Python 3.8+ with required packages
    """)
    
    st.markdown("""
    ### Development Team:
    - [Your Name]
    - [Your Friend's Name]
    
    ### Version: 1.0.0
    """)

# Main app function
def main():
    st.set_page_config(
        page_title="Academic Document Verification",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize database
    init_db()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Verify Document", "History", "Analytics", "About"],
        index=0
    )
    
    # Page routing
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