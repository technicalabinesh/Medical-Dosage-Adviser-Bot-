# 💊 Medicine Dosage Calculator – AI-Powered Medical Assistant

## 🧠 Overview
The **Medicine Dosage Calculator** is an **AI-powered medical assistant** that integrates **IBM Watsonx**, **OCR (Tesseract + OpenCV)**, and **Gradio** to help analyze prescriptions, calculate personalized medicine dosages, and generate AI-based medical explanations.  
This project is intended **for educational and demonstration purposes only** — not for real medical decisions.

---

## 🚀 Key Features

### 🩺 Core Functionalities
- 🔍 **Smart Medicine Search:** Fuzzy matching for medicine names (handles typos and variations).  
- 🧮 **Personalized Dosage Calculator:** Automatically adjusts dosage based on patient age and weight.  
- 🧠 **AI Explanation:** Uses IBM Watsonx to explain medicine purpose, mechanism, and safety.  
- 📄 **PDF Reports:** Generates professional medicine dosage reports automatically.  
- 📦 **Batch Download:** Exports all session prescriptions in one consolidated PDF file.

### 🧾 Prescription Processing
- 📸 **Enhanced OCR:** Reads printed/handwritten prescriptions using Tesseract + OpenCV preprocessing.  
- 🤖 **AI Prescription Explanation:** Analyzes extracted text to identify medicines, dosages, and purposes.  
- ⚡ **Quick Analysis:** Summarizes prescriptions for quick understanding.  
- 🌐 **Multilingual Translation:** Supports Tamil, Telugu, Malayalam, Kannada, Hindi, Marathi, Gujarati, French, German, and Spanish.

### 💬 Chatbot
- 🗣️ **Medical Chatbot:** Ask questions about medicines, health conditions, or general medical information.  
- 🧘 **Empathetic & Safe:** Responds with verified, general information and safety reminders.  
- 🔁 **Context Memory:** Keeps last few exchanges for conversational continuity.

### 🎤 Other Enhancements
- 🎙️ Voice input for medicine name  
- 📦 Downloadable prescription history  
- 🌐 Multi-language translation for AI results  
- 🖼️ Enhanced OCR image preprocessing  
- 🩺 Educational AI explanations with disclaimer  

---

## 🧩 Technologies Used
| Category | Tools / Libraries |
|-----------|------------------|
| Frontend UI | **Gradio** |
| AI Model | **IBM Watsonx.ai** *(Mistral-small-3-1-24b-instruct)* |
| OCR | **Pillow**, **pytesseract**, **OpenCV** |
| Data Handling | **Pandas**, **RapidFuzz**, **Regex** |
| Translation | **Deep Translator (Google)** |
| Report Generation | **ReportLab** |
| Language | **Python 3.10+** |

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Me2. Install Dependencies
pip install gradio pandas ibm-watsonx-ai deep-translator rapidfuzz reportlab pillow pytesseract openpyxl
pip install opencv-python  # optional but recommended

3. Install Tesseract OCR
🪟 Windows:

Download: Tesseract OCR GitHub

Add installation path (e.g., C:\Program Files\Tesseract-OCR) to your System PATH.

🍎 macOS:
brew install tesseract

🐧 Linux:
sudo apt-get install tesseract-ocr

⚙️ How to Run the App
python "Medical Assistant.py"


Then open the Gradio interface in your browser.
You’ll see the following tabs:

🧭 Tabs Overview
Tab	Purpose
🔧 Watsonx Setup	Enter IBM API key & Project ID
📂 Dataset Setup	Upload medicine dataset (CSV/Excel)
💊 Dosage Calculator	Calculate dosage, get AI explanation & PDF
📸 Prescription Analyzer	Upload image → OCR → AI explanation
🤖 Medical Chatbot	Ask any medical or health question
📦 Download History	Export all prescriptions as a PDF
📊 Dataset Format

Upload a .csv or .xlsx file with the following columns:

Column	Description
Name	Medicine name (Required)
Classification	Drug type (Optional)
Indication	Purpose (Optional)
Strength	Dosage strength (Optional)

Example:

Name,Classification,Indication,Strength
Paracetamol,Analgesic,Pain relief,500mg
Amoxicillin,Antibiotic,Bacterial infection,250mg

⚠️ Disclaimer

🧠 For educational and demonstration purposes only.

🩺 Always consult a licensed healthcare professional before using any medication.

⚕️ The AI explanations are not substitutes for professional medical advice.

🧾 OCR accuracy depends on image clarity and language.

🔒 Privacy & Security

✅ All data processed locally within session

✅ No external data storage

✅ Encrypted API credentials

✅ History cleared on app restart

🆕 What’s New (Latest Version)

🤖 Medical Chatbot Tab with context-aware Q&A

📸 AI Prescription Explanation (extracts & explains all medicines)

🎨 Improved UI with emojis, better feedback, and modern layout

📦 Batch PDF Export for complete prescription history

🧑‍💻 Author

Abinesh M.
📧 [Add your email/contact here]
💼 Data Analyst | AI & Python Developer

🩹 Support

If you encounter issues:

Ensure all dependencies are installed correctly

Check Tesseract installation path

Verify IBM Watsonx credentials

Use high-quality prescription images

Re-run after pip install opencv-python for better OCR

🧭 License

This project is licensed under the MIT License — free to use and modify with attribution.


---

Would you like me to **generate this README.md file** automatically and give you a **ready-to-downloadical-Assistant-AI.git
cd Medical-Assistant-AI
