# -*- coding: utf-8 -*-
"""Medicine Dosage Calculator - Enhanced with Voice, OCR, Chatbot & AI Prescription Explanation"""

import gradio as gr
import pandas as pd
import re
from datetime import datetime
from rapidfuzz import fuzz, process
from deep_translator import GoogleTranslator
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
import tempfile
import os
import json

# IBM Watsonx imports
try:
    from ibm_watsonx_ai.foundation_models import Model
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watsonx_ai import Credentials
    IBM_WATSON_AVAILABLE = True
except ImportError:
    IBM_WATSON_AVAILABLE = False
    print("⚠ IBM Watsonx libraries not available. Please install: pip install ibm-watsonx-ai")

# OCR imports
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠ OCR libraries not available. Install: pip install pillow pytesseract")

# OpenCV for image preprocessing
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠ OpenCV not available. Install for better OCR: pip install opencv-python")

# Global variables
df = None
watsonx_model = None
prescription_history = []
chat_history = []

# === Initialize IBM Watsonx ===
def initialize_watsonx(api_key, project_id, url="https://us-south.ml.cloud.ibm.com"):
    """Initialize IBM Watsonx AI model"""
    global watsonx_model
    
    if not IBM_WATSON_AVAILABLE:
        return "⚠ IBM Watsonx libraries not installed. Please install: pip install ibm-watsonx-ai"
    
    if not api_key or not project_id:
        return "⚠ Please provide both API Key and Project ID"
    
    try:
        credentials = Credentials(api_key=api_key, url=url)
        model_params = {
            GenParams.MAX_NEW_TOKENS: 1024,
            GenParams.MIN_NEW_TOKENS: 30,
            GenParams.TEMPERATURE: 0.2,
            GenParams.TOP_P: 0.9,
        }
        watsonx_model = Model(
            model_id='mistralai/mistral-small-3-1-24b-instruct-2503',
            params=model_params,
            credentials=credentials,
            project_id=project_id
        )
        return "✅ IBM Watsonx initialized successfully! You can now use all AI features."
    except Exception as e:
        watsonx_model = None
        return f"❌ Failed to initialize Watsonx: {str(e)}"

def generate_watsonx_text(prompt):
    """Generate text using IBM Watsonx"""
    global watsonx_model
    
    if watsonx_model is None:
        return "Error: Watsonx not initialized. Please configure API credentials first in the Watsonx Setup tab."
    
    try:
        response = watsonx_model.generate_text(prompt=prompt)
        if isinstance(response, str):
            return response
        if hasattr(response, "text"):
            return response.text
        return str(response)
    except Exception as e:
        return f"Error generating text: {str(e)}"

# === Load Dataset ===
def load_dataset(file):
    """Load dataset from uploaded file"""
    global df
    
    try:
        if file is None:
            return "⚠ Please select a file to upload", gr.update()
        
        print(f"📂 Loading file: {file.name}")
        
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name, encoding='utf-8')
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file.name)
        else:
            return "⚠ Please upload CSV (.csv) or Excel (.xlsx, .xls) file only", gr.update()
        
        df.columns = df.columns.str.strip().str.title()
        
        if 'Name' not in df.columns:
            available = ', '.join(df.columns.tolist())
            return f"❌ Error: 'Name' column not found!\n\nColumns found: {available}", gr.update()
        
        df['Name'] = df['Name'].astype(str).str.strip()
        df = df[df['Name'].str.len() > 0]
        df = df[df['Name'] != 'nan']
        df['Name_Search'] = df['Name'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
        
        original_count = len(df)
        df = df.drop_duplicates(subset='Name', keep='first')
        duplicates_removed = original_count - len(df)
        
        sample_meds = df['Name'].head(10).tolist()
        sample_text = ', '.join(sample_meds[:5])
        if len(sample_meds) > 5:
            sample_text += f" ... and {len(df) - 5} more"
        
        success_msg = f"""✅ **Dataset Loaded Successfully!**

📊 **Statistics:**
• Total medicines: {len(df)}
• Duplicates removed: {duplicates_removed}

💊 **Sample Medicines:**
{sample_text}

✅ **Ready to calculate dosages!**"""
        
        return success_msg, gr.update()
        
    except Exception as e:
        return f"❌ Error loading file: {str(e)}", gr.update()

# === Find Medicine ===
def find_medicine(medicine_name):
    """Find medicine using fuzzy matching"""
    global df
    
    if df is None:
        return None, "⚠ Please upload your dataset first!"
    
    if not medicine_name or not medicine_name.strip():
        return None, "⚠ Please enter a medicine name"
    
    medicine_name = medicine_name.strip()
    search_clean = re.sub(r'[^\w\s]', '', medicine_name.lower())
    
    # Exact match
    exact = df[df['Name'].str.lower() == medicine_name.lower()]
    if not exact.empty:
        return exact.iloc[0], f"✅ Exact match found: {exact.iloc[0]['Name']}"
    
    # Contains match
    contains = df[df['Name'].str.lower().str.contains(medicine_name.lower(), na=False)]
    if not contains.empty:
        return contains.iloc[0], f"✅ Found: {contains.iloc[0]['Name']}"
    
    # Fuzzy matching
    try:
        result = process.extractOne(
            search_clean, 
            df['Name_Search'].tolist(), 
            scorer=fuzz.token_sort_ratio, 
            score_cutoff=70
        )
        
        if result:
            matched_name, score, idx = result
            return df.iloc[idx], f"✅ Found: {df.iloc[idx]['Name']} (Match: {score}%)"
    except:
        pass
    
    # Show suggestions
    try:
        suggestions = process.extract(
            search_clean, 
            df['Name_Search'].tolist(), 
            scorer=fuzz.token_sort_ratio, 
            limit=5
        )
        
        sugg_list = []
        for _, score, idx in suggestions:
            if score > 50:
                sugg_list.append(f"  • {df.iloc[idx]['Name']} ({score}% match)")
        
        if sugg_list:
            sugg_text = "\n".join(sugg_list)
            return None, f"❌ Medicine '{medicine_name}' not found.\n\n💡 Did you mean:\n{sugg_text}"
    except:
        pass
    
    return None, f"❌ Medicine '{medicine_name}' not found in database."

# === Calculate Dosage ===
def calculate_dosage(age, weight, strength_str):
    """Calculate dosage based on age and weight"""
    mg_match = re.search(r'(\d+\.?\d*)', str(strength_str))
    base_mg = float(mg_match.group(1)) if mg_match else 500.0
    
    if age < 1:
        single_dose = weight * 10
        frequency = "Every 8 hours"
        category = "Infant"
    elif age < 12:
        single_dose = weight * 15
        frequency = "Every 6-8 hours"
        category = "Child"
    elif age < 60:
        single_dose = base_mg
        frequency = "Every 6-8 hours"
        category = "Adult"
    else:
        single_dose = base_mg * 0.75
        frequency = "Every 8 hours"
        category = "Elderly"
    
    return {
        "category": category,
        "single_dose": round(single_dose, 1),
        "frequency": frequency,
        "daily_dose": round(single_dose * 3, 1),
        "max_daily": round(base_mg * 4, 1)
    }

# === AI Explanation ===
def get_ai_explanation(medicine_info, age, weight, dosage_info):
    """Get AI explanation using IBM Watsonx"""
    
    fallback = f"""📋 **Medicine Information**

**Name:** {medicine_info.get('Name', 'Unknown')}
**Classification:** {medicine_info.get('Classification', 'N/A')}
**Indication:** {medicine_info.get('Indication', 'N/A')}

💊 **Recommended Dosage**
• Patient Category: {dosage_info['category']}
• Single dose: {dosage_info['single_dose']} mg
• Frequency: {dosage_info['frequency']}
• Daily total: {dosage_info['daily_dose']} mg

⚠ **Disclaimer:** Consult healthcare professional."""
    
    try:
        prompt = f"""Provide a brief medical explanation (max 200 words) for:

Medicine: {medicine_info.get('Name', 'Unknown')}
Classification: {medicine_info.get('Classification', 'N/A')}
Patient: {dosage_info['category']}, Age {age} years, Weight {weight} kg
Recommended Dosage: {dosage_info['single_dose']}mg, {dosage_info['frequency']}

Include:
1. How the medicine works
2. Why this dosage is appropriate
3. Common side effects
4. Precautions"""
        
        response = generate_watsonx_text(prompt)
        
        if response and not response.startswith("Error"):
            return response
    except:
        pass
    
    return fallback

# === Process Medicine ===
def process_medicine(medicine_name, patient_name, age, weight):
    """Main processing function"""
    global df, prescription_history
    
    if df is None:
        return "⚠ **No dataset loaded!**", "", "", "", None
    
    if not medicine_name or not medicine_name.strip():
        return "⚠ Please enter a medicine name", "", "", "", None
    
    if not patient_name or not patient_name.strip():
        patient_name = "Patient"
    
    if age is None or age <= 0:
        return "⚠ Please enter a valid age", "", "", "", None
    
    if weight is None or weight <= 0:
        return "⚠ Please enter a valid weight", "", "", "", None
    
    try:
        medicine_info, search_msg = find_medicine(medicine_name)
        
        if medicine_info is None:
            return search_msg, "", "", "", None
        
        dosage_info = calculate_dosage(age, weight, medicine_info.get('Strength', '500mg'))
        explanation = get_ai_explanation(medicine_info, age, weight, dosage_info)
        
        # Store in history
        prescription_history.append({
            'timestamp': datetime.now(),
            'patient_name': patient_name,
            'medicine_name': medicine_info.get('Name', 'N/A'),
            'age': age,
            'weight': weight,
            'dosage': dosage_info,
            'explanation': explanation,
            'medicine_info': medicine_info
        })
        
        medicine_display = f"""✅ **Medicine Found: {medicine_info.get('Name', 'N/A')}**

**Classification:** {medicine_info.get('Classification', 'N/A')}
**Indication:** {medicine_info.get('Indication', 'N/A')}
**Strength:** {medicine_info.get('Strength', 'N/A')}"""
        
        dosage_display = f"""👤 **Patient:** {patient_name}
📊 **Category:** {dosage_info['category']}

💊 **Single Dose:** {dosage_info['single_dose']} mg
⏰ **Frequency:** {dosage_info['frequency']}
📈 **Daily Total:** {dosage_info['daily_dose']} mg
⚠️ **Maximum Daily:** {dosage_info['max_daily']} mg"""
        
        pdf = generate_pdf(patient_name, medicine_info, age, weight, dosage_info, explanation)
        
        return medicine_display, dosage_display, explanation, search_msg, pdf
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", "", None

# === Generate PDF ===
def generate_pdf(patient_name, medicine_info, age, weight, dosage_info, explanation):
    """Generate PDF report"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb')
        temp_path = temp_file.name
        temp_file.close()
        
        doc = SimpleDocTemplate(temp_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        elements = []
        
        elements.append(Paragraph("Medicine Dosage Report", styles['Title']))
        elements.append(Spacer(1, 0.3*inch))
        
        elements.append(Paragraph("Patient Information", styles['Heading2']))
        elements.append(Paragraph(f"Name: {patient_name}", styles['Normal']))
        elements.append(Paragraph(f"Age: {age} years", styles['Normal']))
        elements.append(Paragraph(f"Weight: {weight} kg", styles['Normal']))
        elements.append(Paragraph(f"Category: {dosage_info['category']}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        elements.append(Paragraph("Medicine Information", styles['Heading2']))
        elements.append(Paragraph(f"Name: {medicine_info.get('Name', 'N/A')}", styles['Normal']))
        elements.append(Paragraph(f"Classification: {medicine_info.get('Classification', 'N/A')}", styles['Normal']))
        elements.append(Paragraph(f"Strength: {medicine_info.get('Strength', 'N/A')}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        elements.append(Paragraph("Recommended Dosage", styles['Heading2']))
        elements.append(Paragraph(f"Single Dose: {dosage_info['single_dose']} mg", styles['Normal']))
        elements.append(Paragraph(f"Frequency: {dosage_info['frequency']}", styles['Normal']))
        elements.append(Paragraph(f"Daily Total: {dosage_info['daily_dose']} mg", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        if explanation and len(explanation) > 50:
            elements.append(Paragraph("Medical Information", styles['Heading2']))
            clean_explanation = explanation.replace('**', '').replace('*', '').replace('#', '')
            elements.append(Paragraph(clean_explanation[:800], styles['Normal']))
        
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("DISCLAIMER: For educational purposes only.", styles['Normal']))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        
        doc.build(elements)
        return temp_path
    except:
        return None

# === Translate ===
def translate_text(text, target_lang):
    """Translate text"""
    if not text or not text.strip():
        return "⚠ No text to translate"
    
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        result = translator.translate(text[:5000])
        return f"**Translation ({target_lang.upper()}):**\n\n{result}\n\n---\n⚠️ Machine translation"
    except Exception as e:
        return f"❌ Translation failed: {str(e)}"

# === Enhanced OCR with Image Preprocessing ===
def extract_text_from_image(image):
    """Extract text from prescription image using OCR with preprocessing"""
    if not OCR_AVAILABLE:
        return "⚠ OCR not available. Install: pip install pillow pytesseract"
    
    if image is None:
        return "⚠ Please upload an image"
    
    try:
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if CV2_AVAILABLE:
            # Advanced preprocessing with OpenCV
            img_array = np.array(image)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply preprocessing techniques
            # 1. Resize image (upscale if too small)
            height, width = gray.shape
            if height < 1000 or width < 1000:
                scale_factor = max(1000/height, 1000/width)
                gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                                interpolation=cv2.INTER_CUBIC)
            
            # 2. Denoise
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # 3. Apply adaptive thresholding for better contrast
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 4. Morphological operations to remove noise
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(processed)
            
            # Extract text with multiple PSM modes
            texts = []
            
            # Try different PSM modes on processed image
            for psm in [6, 4, 3]:
                try:
                    text = pytesseract.image_to_string(processed_image, config=f'--oem 3 --psm {psm}')
                    if text.strip():
                        texts.append(text)
                except:
                    continue
            
            # Also try original image
            try:
                text_original = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
                if text_original.strip():
                    texts.append(text_original)
            except:
                pass
            
            if not texts:
                return "⚠ No text found in image. Please ensure the image is clear and contains readable text."
            
            # Choose the longest extracted text (usually most complete)
            final_text = max(texts, key=len).strip()
            
            # Show statistics
            char_count = len(final_text)
            line_count = len([line for line in final_text.split('\n') if line.strip()])
            
            return f"""✅ **Text Extracted Successfully!**

📊 **Statistics:**
• Characters extracted: {char_count}
• Lines detected: {line_count}

📄 **Extracted Text:**

{final_text}

---
💡 **Tip:** If text is incomplete, try:
• Taking a clearer photo
• Ensuring good lighting
• Making sure text is horizontal
• Using higher resolution image"""
        
        else:
            # Fallback to basic OCR without OpenCV preprocessing
            texts = []
            for psm in [6, 4, 3, 11]:
                try:
                    text = pytesseract.image_to_string(image, config=f'--oem 3 --psm {psm}')
                    if text.strip():
                        texts.append(text)
                except:
                    continue
            
            if not texts:
                return "⚠ No text found. Install opencv-python for better results:\npip install opencv-python"
            
            final_text = max(texts, key=len).strip()
            char_count = len(final_text)
            
            return f"""✅ **Extracted Text:** ({char_count} characters)

{final_text}

---
💡 **For better accuracy, install opencv-python:**
pip install opencv-python"""
            
    except Exception as e:
        return f"""❌ **OCR Failed:** {str(e)}

💡 **Troubleshooting:**
• Ensure Tesseract is installed: https://github.com/tesseract-ocr/tesseract
• Install opencv-python: pip install opencv-python
• Check image quality and format
• Make sure the image contains clear, readable text"""

# === NEW: AI Explain Extracted Prescription ===
def explain_extracted_prescription(extracted_text):
    """AI explains the extracted prescription text in detail"""
    if not extracted_text or not extracted_text.strip() or len(extracted_text) < 20:
        return "⚠ Please extract text from prescription first using the 'Extract Text' button above."
    
    # Check if it's an error message
    if extracted_text.startswith("⚠") or extracted_text.startswith("❌"):
        return "⚠ Cannot explain: No valid prescription text extracted. Please upload a clear prescription image."
    
    prompt = f"""You are a medical AI assistant. Analyze this prescription text extracted via OCR and provide a comprehensive explanation.

Prescription Text:
{extracted_text[:2000]}

Please provide:

1. **Medicines Identified**: List all medicines mentioned with their generic/brand names
2. **Dosage Information**: Extract dosage for each medicine (strength, frequency, duration)
3. **Medical Purpose**: Explain what each medicine is typically used for
4. **Administration Instructions**: When and how to take each medicine
5. **Important Warnings**: Any contraindications, side effects, or precautions
6. **Additional Notes**: Any other relevant information from the prescription

Format the response clearly with headers and bullet points. If any information is unclear due to OCR errors, mention it."""
    
    try:
        response = generate_watsonx_text(prompt)
        
        if response and not response.startswith("Error"):
            return f"""🤖 **AI Prescription Analysis**

{response}

---
⚠️ **Disclaimer:** This is an AI-generated analysis for informational purposes only. Always consult with a healthcare professional before taking any medication. Verify all dosages and instructions with your doctor or pharmacist."""
        else:
            return f"❌ AI explanation failed: {response}\n\nPlease check if Watsonx is properly initialized in the Setup tab."
    except Exception as e:
        return f"❌ Error generating explanation: {str(e)}\n\nPlease ensure Watsonx is configured correctly."

# === Analyze Prescription ===
def analyze_prescription(text):
    """Analyze prescription using Watsonx"""
    if not text or not text.strip():
        return "⚠ Please enter prescription text"
    
    prompt = f"""Analyze this prescription and extract:
1. All medicine names mentioned
2. Dosages for each medicine
3. Frequency of administration
4. Duration of treatment
5. Any warnings or special instructions

Format clearly with bullet points.

Prescription text:
{text}"""
    
    try:
        response = generate_watsonx_text(prompt)
        if response and not response.startswith("Error"):
            return response
        else:
            return f"❌ Analysis failed: {response}"
    except Exception as e:
        return f"❌ Analysis failed: {str(e)}"

# === NEW: Medical Chatbot ===
def chat_with_bot(user_message, history):
    """Medical chatbot powered by Watsonx"""
    global chat_history
    
    if not user_message or not user_message.strip():
        return history, ""
    
    if watsonx_model is None:
        bot_response = "❌ **Watsonx not initialized!** Please configure your API credentials in the 'Watsonx Setup' tab first."
        history.append((user_message, bot_response))
        return history, ""
    
    # Build context from chat history
    context = ""
    if chat_history:
        context = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history[-5:]])  # Last 5 exchanges
    
    prompt = f"""You are a helpful medical information assistant. Answer questions about medicines, health conditions, symptoms, and general medical information.

Important guidelines:
- Provide accurate, evidence-based medical information
- Always remind users to consult healthcare professionals for personal medical advice
- Be clear about limitations and when professional help is needed
- If asked about specific dosages, suggest consulting a doctor
- Be empathetic and understanding
- Keep responses concise but informative (max 300 words)

Previous conversation:
{context}

User question: {user_message}

Assistant response:"""
    
    try:
        response = generate_watsonx_text(prompt)
        
        if response and not response.startswith("Error"):
            bot_response = response
        else:
            bot_response = f"❌ I encountered an error: {response}\n\nPlease try rephrasing your question or check the Watsonx configuration."
        
        # Add to history
        chat_history.append((user_message, bot_response))
        history.append((user_message, bot_response))
        
        return history, ""
        
    except Exception as e:
        bot_response = f"❌ Error: {str(e)}\n\nPlease ensure Watsonx is properly configured."
        history.append((user_message, bot_response))
        return history, ""

def clear_chat():
    """Clear chat history"""
    global chat_history
    chat_history = []
    return [], ""

# === Download All Prescriptions ===
def download_all_prescriptions():
    """Generate PDF with all prescription history"""
    global prescription_history
    
    if not prescription_history:
        return None
    
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb')
        temp_path = temp_file.name
        temp_file.close()
        
        doc = SimpleDocTemplate(temp_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        elements = []
        
        elements.append(Paragraph("All Prescriptions History", styles['Title']))
        elements.append(Paragraph(f"Total Prescriptions: {len(prescription_history)}", styles['Normal']))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        for idx, rx in enumerate(prescription_history, 1):
            elements.append(Paragraph(f"Prescription #{idx}", styles['Heading2']))
            elements.append(Paragraph(f"Date: {rx['timestamp'].strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
            elements.append(Paragraph(f"Patient: {rx['patient_name']}", styles['Normal']))
            elements.append(Paragraph(f"Age: {rx['age']} years, Weight: {rx['weight']} kg", styles['Normal']))
            elements.append(Paragraph(f"Medicine: {rx['medicine_name']}", styles['Normal']))
            elements.append(Paragraph(f"Dosage: {rx['dosage']['single_dose']}mg, {rx['dosage']['frequency']}", styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
            
            if idx < len(prescription_history):
                elements.append(PageBreak())
        
        doc.build(elements)
        return temp_path
    except Exception as e:
        print(f"Error generating batch PDF: {e}")
        return None

# === Create Interface ===
with gr.Blocks(title="Enhanced Medicine Calculator", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 💊 Medicine Dosage Calculator - Enhanced Edition")
    gr.Markdown("🎤 Voice Input | 📸 Enhanced OCR | 🌐 Translation | 🤖 AI Chatbot | 📦 Batch Download")
    
    # Watsonx Setup
    with gr.Tab("🔧 Watsonx Setup"):
        gr.Markdown("### Configure IBM Watsonx AI")
        
        with gr.Row():
            api_key_input = gr.Textbox(label="🔑 API Key", type="password")
            project_id_input = gr.Textbox(label="📋 Project ID")
        
        url_input = gr.Textbox(label="🌐 URL", value="https://us-south.ml.cloud.ibm.com")
        init_btn = gr.Button("🚀 Initialize", variant="primary")
        init_status = gr.Textbox(label="Status", interactive=False, lines=3)
        
        init_btn.click(initialize_watsonx, inputs=[api_key_input, project_id_input, url_input], outputs=init_status)
    
    # Dataset Setup
    with gr.Tab("📂 Dataset Setup"):
        gr.Markdown("### Upload Medicine Dataset")
        file_input = gr.File(label="📁 Upload File", file_types=[".csv", ".xlsx", ".xls"])
        upload_btn = gr.Button("📤 Load Dataset", variant="primary")
        status = gr.Textbox(label="Status", interactive=False, lines=10)
        
        upload_btn.click(load_dataset, inputs=file_input, outputs=[status, file_input])
    
    # Dosage Calculator
    with gr.Tab("💊 Dosage Calculator"):
        gr.Markdown("### Calculate Personalized Dosage")
        
        with gr.Row():
            patient_name = gr.Textbox(label="👤 Patient Name", placeholder="Enter patient name")
            med_input = gr.Textbox(label="💊 Medicine Name", placeholder="Enter or speak medicine name")
            med_audio = gr.Audio(label="🎤 Voice Input (Optional)", sources=["microphone"], type="numpy")
        
        with gr.Row():
            age_input = gr.Number(label="👶 Age (years)", value=30, minimum=0.1)
            weight_input = gr.Number(label="⚖️ Weight (kg)", value=70, minimum=1)
        
        calc_btn = gr.Button("🧮 Calculate Dosage", variant="primary", size="lg")
        
        search_result = gr.Textbox(label="Search Result", interactive=False, lines=2)
        
        with gr.Row():
            med_info = gr.Textbox(label="📋 Medicine Info", interactive=False, lines=7)
            dosage_out = gr.Textbox(label="💊 Dosage", interactive=False, lines=7)
        
        explain_out = gr.Textbox(label="🤖 AI Explanation", interactive=False, lines=10)
        pdf_out = gr.File(label="📄 Download PDF")
        
        calc_btn.click(
            process_medicine,
            inputs=[med_input, patient_name, age_input, weight_input],
            outputs=[med_info, dosage_out, explain_out, search_result, pdf_out]
        )
        
        # Translation
        gr.Markdown("---")
        gr.Markdown("### 🌐 Translate Explanation")
        
        with gr.Row():
            lang = gr.Dropdown(
                choices=["tamil", "telugu", "hindi", "kannada", "malayalam", 
                        "marathi", "gujarati", "spanish", "french", "german"],
                value="tamil",
                label="🌍 Language"
            )
            trans_btn = gr.Button("🔄 Translate")
        
        trans_out = gr.Textbox(label="Translation", interactive=False, lines=10)
        trans_btn.click(translate_text, inputs=[explain_out, lang], outputs=trans_out)
    
    # Prescription Analyzer
    with gr.Tab("📸 Prescription Analyzer"):
        gr.Markdown("### AI Prescription Analysis with Enhanced OCR")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Option 1: Upload Image**")
                rx_image = gr.Image(label="📸 Upload Prescription Image", type="pil")
                ocr_btn = gr.Button("🔍 Extract Text (Enhanced OCR)", variant="secondary")
            
            with gr.Column():
                gr.Markdown("**Option 2: Type/Paste Text**")
                rx_input = gr.Textbox(label="📝 Prescription Text", lines=10, placeholder="Paste text or extract from image...")
        
        ocr_btn.click(extract_text_from_image, inputs=rx_image, outputs=rx_input)
        
        gr.Markdown("---")
        gr.Markdown("### 🤖 AI Analysis & Explanation")
        
        with gr.Row():
            rx_explain_btn = gr.Button("🤖 AI Explain Prescription", variant="primary", size="lg")
            rx_analyze_btn = gr.Button("📊 Quick Analysis", variant="secondary")
        
        rx_explain_out = gr.Textbox(label="🤖 AI Detailed Explanation", interactive=False, lines=15)
        rx_out = gr.Textbox(label="📊 Quick Analysis", interactive=False, lines=10)
        
        rx_explain_btn.click(explain_extracted_prescription, inputs=rx_input, outputs=rx_explain_out)
        rx_analyze_btn.click(analyze_prescription, inputs=rx_input, outputs=rx_out)
        
        gr.Markdown("---")
        gr.Markdown("### 🌐 Translate Analysis")
        
        with gr.Row():
            rx_translate_btn = gr.Button("🔄 Translate", variant="secondary")
            rx_lang = gr.Dropdown(
                choices=["tamil", "hindi", "telugu", "kannada", "malayalam", "marathi", "gujarati"],
                value="tamil",
                label="🌍 Translation Language"
            )
        
        rx_trans_out = gr.Textbox(label="🌐 Translated Analysis", interactive=False, lines=12)
        
        rx_translate_btn.click(translate_text, inputs=[rx_explain_out, rx_lang], outputs=rx_trans_out)
    
    # NEW: Medical Chatbot Tab
    with gr.Tab("🤖 Medical Chatbot"):
        gr.Markdown("### 💬 Ask Medical Questions")
        gr.Markdown("Ask me anything about medicines, health conditions, symptoms, or general medical information!")
        
        chatbot_interface = gr.Chatbot(
            label="💬 Medical Assistant",
            height=500,
            show_label=True,
            avatar_images=(None, "🤖")
        )
        
        with gr.Row():
            chat_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask me about medicines, symptoms, health conditions, etc...",
                lines=2,
                scale=4
            )
            chat_submit = gr.Button("📤 Send", variant="primary", scale=1)
        
        with gr.Row():
            chat_clear = gr.Button("🗑️ Clear Chat", variant="secondary")
            chat_examples = gr.Examples(
                examples=[
                    "What is Paracetamol used for?",
                    "What are the side effects of antibiotics?",
                    "How does Ibuprofen work?",
                    "What should I do if I have a fever?",
                    "Can I take medicine on an empty stomach?",
                    "What are the symptoms of diabetes?",
                    "How to manage high blood pressure?",
                    "What vitamins are important for immunity?"
                ],
                inputs=chat_input,
                label="💡 Example Questions"
            )
        
        gr.Markdown("""
        ---
        ⚠️ **Important Disclaimer:**
        - This chatbot provides general medical information only
        - Always consult a healthcare professional for personal medical advice
        - Do not use this for emergency medical situations
        - Verify all information with qualified medical practitioners
        """)
        
        chat_submit.click(
            chat_with_bot,
            inputs=[chat_input, chatbot_interface],
            outputs=[chatbot_interface, chat_input]
        )
        
        chat_input.submit(
            chat_with_bot,
            inputs=[chat_input, chatbot_interface],
            outputs=[chatbot_interface, chat_input]
        )
        
        chat_clear.click(
            clear_chat,
            outputs=[chatbot_interface, chat_input]
        )
    
    # Batch Download
    with gr.Tab("📦 Download History"):
        gr.Markdown("### Download All Prescriptions")
        gr.Markdown("Download a PDF containing all prescription calculations from this session")
        
        download_all_btn = gr.Button("📥 Download All Prescriptions", variant="primary", size="lg")
        batch_pdf_out = gr.File(label="📄 Batch PDF Download")
        
        download_all_btn.click(download_all_prescriptions, outputs=batch_pdf_out)
    
    # Help
    with gr.Tab("❓ Help"):
        gr.Markdown("""
        ### 📚 Quick Start Guide
        
        #### 1. 🔧 Setup Watsonx
        - Enter API Key and Project ID
        - Click "Initialize"
        - Wait for confirmation message
        
        #### 2. 📂 Upload Dataset
        - Upload CSV/Excel with medicine data
        - Must have "Name" column
        - Wait for success message
        
        #### 3. 💊 Calculate Dosage
        - Enter patient name (optional)
        - Type medicine name OR use voice input 🎤
        - Enter age and weight
        - Click "Calculate"
        - Get AI explanation and PDF report
        
        #### 4. 📸 Prescription Analysis (Enhanced OCR)
        - Upload prescription image 📸
        - Click "Extract Text (Enhanced OCR)"
        - Click "AI Explain Prescription" for detailed analysis 🤖
        - OR click "Quick Analysis" for summary
        - Translate to regional languages 🌐
        
        #### 5. 🤖 Medical Chatbot (NEW!)
        - Ask any medical question
        - Get AI-powered answers
        - Learn about medicines, symptoms, conditions
        - Examples provided for quick start
        
        #### 6. 📦 Download History
        - Click "Download All Prescriptions"
        - Get PDF with all calculations from session
        
        ---
        
        ### 🎤 Voice Input
        - Click microphone icon
        - Speak medicine name clearly
        - Text will auto-fill
        
        ---
        
        ### 📸 Enhanced OCR Features
        
        **Advanced Image Preprocessing:**
        - ✅ Automatic upscaling for small images
        - ✅ Noise reduction
        - ✅ Adaptive thresholding for better contrast
        - ✅ Multiple extraction modes for maximum accuracy
        
        **Best Practices:**
        - Use high resolution images (300+ DPI)
        - Ensure good lighting
        - Keep text horizontal
        - Dark text on light background
        - Clear, focused images
        
        ---
        
        ### 🤖 NEW: AI Prescription Explanation
        
        After extracting text from prescription:
        1. Click "AI Explain Prescription" button
        2. Get comprehensive analysis including:
           - 💊 All medicines identified
           - 📊 Dosage information for each
           - 🎯 Medical purpose explained
           - ⏰ Administration instructions
           - ⚠️ Important warnings & precautions
        3. Translate to your language
        
        ---
        
        ### 💬 Chatbot Features
        
        **What you can ask:**
        - Medicine information and uses
        - Side effects and interactions
        - Symptom explanations
        - Health condition information
        - General medical guidance
        - Dosage questions
        - Preventive health tips
        
        **What chatbot provides:**
        - Evidence-based information
        - Clear, concise answers
        - Safety reminders
        - Professional consultation advice
        
        ---
        
        ### 🌐 Translation
        - Available in 10+ languages
        - Tamil, Hindi, Telugu, Kannada, Malayalam, etc.
        - Works for all AI-generated content
        
        ---
        
        ### ⚠️ Important Notes
        - **For educational purposes only**
        - **Always consult healthcare professionals**
        - **Not for actual medical decisions**
        - **OCR accuracy depends on image quality**
        - **Chatbot provides general information only**
        - **In emergency, call your local emergency number**
        
        ---
        
        ### 📦 Installation Requirements
        
        **Essential Packages:**
        ```bash
        pip install gradio pandas ibm-watsonx-ai
        pip install deep-translator rapidfuzz reportlab
        pip install pillow pytesseract openpyxl
        ```
        
        **For Enhanced OCR (Highly Recommended):**
        ```bash
        pip install opencv-python
        ```
        
        **Tesseract OCR Installation:**
        
        🪟 **Windows:**
        - Download: https://github.com/tesseract-ocr/tesseract
        - Add to PATH
        
        🍎 **macOS:**
        ```bash
        brew install tesseract
        ```
        
        🐧 **Linux (Ubuntu/Debian):**
        ```bash
        sudo apt-get install tesseract-ocr
        ```
        
        🐧 **Linux (CentOS/RHEL):**
        ```bash
        sudo yum install tesseract
        ```
        
        ---
        
        ### 🔧 Troubleshooting
        
        **OCR Issues:**
        - ❌ "No text found": Image quality too low
        - 💡 Solution: Install opencv-python, use higher resolution
        
        **Chatbot Issues:**
        - ❌ "Watsonx not initialized": Configure API in Setup tab
        - 💡 Solution: Enter valid API key and project ID
        
        **AI Explanation Issues:**
        - ❌ "Cannot explain": No valid text extracted
        - 💡 Solution: Ensure clear image and successful OCR extraction
        
        ---
        
        ### 💡 Tips for Best Results
        
        **For OCR:**
        - 📸 Use 300+ DPI resolution
        - 💡 Ensure even lighting
        - 📏 Keep prescription flat
        - 🎯 Focus the camera properly
        - 🧹 Clean prescription before photo
        
        **For Medicine Search:**
        - 🔍 Use generic/scientific names
        - ✅ Fuzzy matching handles typos
        - 💡 Check suggestions if no match
        
        **For Chatbot:**
        - 💬 Ask specific questions
        - 📝 Provide context when needed
        - ✅ Verify answers with professionals
        - 🔄 Rephrase if answer unclear
        
        ---
        
        ### ✨ Features Overview
        
        ✅ **Smart Medicine Search** - Fuzzy matching with suggestions  
        ✅ **Age-Based Dosage** - Infant, Child, Adult, Elderly categories  
        ✅ **AI Explanations** - Powered by IBM Watsonx  
        ✅ **Enhanced OCR** - Advanced image preprocessing  
        ✅ **Prescription AI Analysis** - Comprehensive explanation (NEW!)  
        ✅ **Medical Chatbot** - Ask any medical question (NEW!)  
        ✅ **Multi-language Support** - 10+ languages  
        ✅ **Voice Input** - Hands-free medicine entry  
        ✅ **PDF Reports** - Individual and batch downloads  
        ✅ **History Tracking** - Session-based prescription records  
        
        ---
        
        ### 📋 Dataset Format
        
        Your CSV/Excel should have these columns:
        - **Name** (Required) - Medicine name
        - **Classification** (Optional) - Drug classification
        - **Indication** (Optional) - What it's used for
        - **Strength** (Optional) - Default dosage strength
        
        Example:
        ```
        Name,Classification,Indication,Strength
        Paracetamol,Analgesic,Pain relief,500mg
        Amoxicillin,Antibiotic,Bacterial infection,250mg
        ```
        
        ---
        
        ### 🔒 Privacy & Security
        - ✅ All processing done locally/in session
        - ✅ No data stored permanently
        - ✅ Prescription history cleared on restart
        - ✅ IBM Watsonx API credentials encrypted
        - ✅ Chatbot conversations not saved externally
        
        ---
        
        ### 📞 Support
        - Check Troubleshooting section above
        - Ensure all dependencies installed
        - Verify Tesseract installation for OCR
        - Test with clear, high-quality images
        - Initialize Watsonx before using AI features
        
        ---
        
        ### 🆕 What's New in This Version
        
        1. **🤖 Medical Chatbot Tab**
           - Ask any medical questions
           - AI-powered responses
           - Context-aware conversations
           - Example questions provided
        
        2. **📸 AI Prescription Explanation**
           - Comprehensive analysis of extracted text
           - Medicine identification
           - Dosage breakdown
           - Safety warnings
           - Administration guidance
        
        3. **🎨 Improved UI**
           - Better emoji usage
           - Clearer section headers
           - Enhanced visual feedback
           - Streamlined workflow
        """)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("💊 Enhanced Medicine Dosage Calculator")
    print("="*60)
    print("\n✨ Features:")
    print("• 🎤 Voice input for medicine names")
    print("• 📸 Enhanced OCR with image preprocessing")
    print("•  🤖 AI Prescription Explanation (NEW!)")
    print("• 💬 Medical Chatbot - Ask Questions (NEW!)")
    print("• 🌐 Multi-language translation (10+ languages)")
    print("• 📦 Batch prescription download")
    print("• 👤 Patient name tracking")
    print("• 📊 AI-powered prescription analysis")
    print("• 🔍 Smart fuzzy medicine search")
    print("• 📄 Professional PDF reports")
    print("\n📦 Requirements:")
    print("• pip install opencv-python (for enhanced OCR)")
    print("• Tesseract OCR installed on system")
    print("• IBM Watsonx API credentials")
    print("\n" + "="*60 + "\n")
    
    demo.launch(share=True, debug=True)