# ============================================================
# CLINICAL NLP
# AUTOMATED MEDICAL REPORT SUMMARIZATION
# Complete Streamlit Application (Environment Protected Version)
# ============================================================

import streamlit as st
import joblib
import re
import nltk
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Clinical NLP Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SAFE TRANSFORMERS & TORCH IMPORT
# ============================================================
# Hugging Face scans all image/vision models on import. If torchvision is missing,
# it throws a ModuleNotFoundError. We catch it here so the UI still renders.
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from nltk.tokenize import sent_tokenize
    transformers_error_msg = None
    dependencies_installed = True
except ModuleNotFoundError as e:
    dependencies_installed = False
    transformers_error_msg = f"Missing dependency: {str(e)}. Please run 'pip install torchvision torch' in your terminal."
except Exception as e:
    dependencies_installed = False
    transformers_error_msg = f"Initialization Error: {str(e)}"

# ============================================================
# NLTK SETUP
# ============================================================
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass

if dependencies_installed:
    setup_nltk()

# ============================================================
# LOAD CONDITION PREDICTION MODEL
# ============================================================
@st.cache_resource
def load_condition_model():
    model_path = "condition_model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"'{model_path}' was not found in the current directory.")

    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"'{vectorizer_path}' was not found in the current directory.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Try loading traditional ML models
try:
    condition_model, tfidf_vectorizer = load_condition_model()
    condition_model_loaded = True
    condition_model_error = None
except Exception as e:
    condition_model_loaded = False
    condition_model = None
    tfidf_vectorizer = None
    condition_model_error = str(e)

# ============================================================
# LOAD AI SUMMARIZATION MODEL
# ============================================================
@st.cache_resource
def load_summarization_model():
    if not dependencies_installed:
        raise RuntimeError(transformers_error_msg)
        
    model_name = "sshleifer/distilbart-xsum-6-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

# Try loading Deep Learning summarizer
try:
    if dependencies_installed:
        tokenizer, summarization_model = load_summarization_model()
        summarizer_loaded = True
        summarizer_error = None
    else:
        summarizer_loaded = False
        summarizer_error = transformers_error_msg
except Exception as e:
    tokenizer = None
    summarization_model = None
    summarizer_loaded = False
    summarizer_error = str(e)

# ============================================================
# HELPER FUNCTIONS (TEXT CLEANING & FINDINGS EXTRACTION)
# ============================================================
def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,;:()\-/%]", "", text)
    return text.strip()

clinical_keywords = [
    "diagnosed", "diagnosis", "assessment", "condition",
    "symptom", "symptoms", "pain", "fever", "headache", "fatigue",
    "nausea", "vomiting", "swelling", "weakness", "dizziness", "cough",
    "improved", "improvement", "decreased", "reduced", "increased", "worsened", "stable",
    "treatment", "therapy", "medication", "prescribed", "administered",
    "follow-up", "follow up", "response", "positive", "negative", "normal",
    "abnormal", "elevated", "low", "high", "severity", "blood pressure", "heart rate"
]

def extract_key_findings(report, max_findings=5):
    try:
        sentences = sent_tokenize(str(report))
    except Exception:
        sentences = re.split(r"(?<=[.!?])\s+", str(report))

    findings = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in clinical_keywords):
            findings.append(sentence.strip())

    findings = list(dict.fromkeys(findings))
    findings = findings[:max_findings]

    if len(findings) == 0:
        if len(sentences) > 0:
            return sentences[:3]
        return ["No significant clinical findings were detected."]
    return findings

def predict_condition(report):
    if not condition_model_loaded:
        return "Condition prediction model is not available."
    try:
        cleaned_report = clean_text(report)
        vectorized_report = tfidf_vectorizer.transform([cleaned_report])
        prediction = condition_model.predict(vectorized_report)[0]
        return str(prediction)
    except Exception as e:
        return f"Unable to predict condition: {str(e)}"

def generate_summary(report):
    if not summarizer_loaded:
        return "The AI summarization model could not be loaded."

    report = clean_text(report)
    word_count = len(report.split())

    if word_count < 30:
        return report

    try:
        inputs = tokenizer(report, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            summary_ids = summarization_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=100,
                min_length=20,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"Unable to generate summary: {str(e)}"

# ============================================================
# DATA & DICTIONARIES
# ============================================================
sample_reports = {
    "Migraine Report": "The patient presented with persistent migraine headaches occurring three to four times per week. Pain severity was rated 8/10. The patient also reported nausea and sensitivity to light. The patient was diagnosed with chronic migraine. Treatment was initiated with prescribed medication. During follow-up, headache frequency decreased significantly and the patient reported improvement in symptoms.",
    "Diabetes Report": "The patient reported increased thirst, frequent urination and persistent fatigue. Blood glucose levels were elevated during laboratory assessment. Further testing showed abnormal fasting glucose levels. The patient was diagnosed with type 2 diabetes mellitus. Treatment was initiated with prescribed medication, dietary modifications and regular physical activity.",
    "Hypertension Report": "The patient presented with elevated blood pressure and occasional headaches. Blood pressure was measured at 160/95 mmHg during clinical assessment. The patient was diagnosed with hypertension. Medication was prescribed along with dietary and lifestyle modifications. During follow-up, blood pressure decreased significantly.",
    "Asthma Report": "The patient reported recurrent shortness of breath, coughing and wheezing. Symptoms increased during physical activity and exposure to dust. Clinical assessment indicated asthma. Inhalation therapy and prescribed medication were initiated.",
    "Custom Report": "Enter or paste your own medical report here."
}

# ============================================================
# UI STYLING & MARKDOWN HEADERS
# ============================================================
st.markdown("""
<style>
.main-title { text-align: center; font-size: 45px; font-weight: 700; margin-bottom: 5px; }
.sub-title { text-align: center; font-size: 18px; color: #888888; margin-bottom: 35px; }
.custom-footer { text-align: center; color: #888888; font-size: 14px; padding-top: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏥 Clinical NLP Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated Medical Report Summarization and Key Findings Extraction</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR STATUS PANEL
# ============================================================
with st.sidebar:
    st.title("🏥 Clinical NLP")
    st.write("This application uses Machine Learning and Natural Language Processing to analyze medical reports.")
    st.divider()
    
    st.subheader("✨ Features")
    st.write("🧹 Medical Text Preprocessing\n\n🩺 Condition Prediction\n\n🔑 Key Findings Extraction\n\n📝 AI-Generated Summary")
    st.divider()
    
    st.subheader("🤖 Model Status")
    
    if condition_model_loaded:
        st.success("✅ Condition Model: Ready")
    else:
        st.error("❌ Condition Model: Not Loaded")
        with st.expander("View Model Error"):
            st.code(condition_model_error)

    if summarizer_loaded:
        st.success("✅ AI Summarizer: Ready")
    else:
        st.error("❌ AI Summarizer: Not Loaded")
        with st.expander("View Summarizer Error"):
            st.code(summarizer_error)
            
    st.divider()
    st.warning("⚠️ Educational and research use only. This application does not replace professional medical advice.")

# ============================================================
# APP INTERACTION LOGIC
# ============================================================
st.subheader("📄 Medical Report Analysis")

selected_sample = st.selectbox("Choose an example medical report:", options=list(sample_reports.keys()))
medical_report = st.text_area("Edit the report or paste your own medical report:", value=sample_reports[selected_sample], height=280)

if medical_report:
    word_count = len(medical_report.split())
    character_count = len(medical_report)
    sentences = re.split(r"[.!?]+", medical_report)
    sentence_count = len([s for s in sentences if s.strip()])

    col1, col2, col3 = st.columns(3)
    col1.metric("📝 Words", word_count)
    col2.metric("🔤 Characters", character_count)
    col3.metric("📄 Sentences", sentence_count)

analyze_button = st.button("🔍 Analyze Medical Report", type="primary", use_container_width=True)

if analyze_button:
    if not medical_report.strip():
        st.warning("Please enter a medical report before clicking Analyze.")
    elif selected_sample == "Custom Report" and medical_report.strip() == sample_reports["Custom Report"]:
        st.warning("Please replace the sample text with your own medical report.")
    else:
        cleaned_report = clean_text(medical_report)
        predicted_cond = predict_condition(cleaned_report)
        key_findings = extract_key_findings(cleaned_report)

        with st.spinner("🤖 Analyzing and summarizing the medical report..."):
            generated_summary = generate_summary(cleaned_report)

        st.success("✅ Medical report analysis completed!")
        st.divider()

        st.subheader("🩺 Predicted Medical Condition")
        if condition_model_loaded:
            st.info(predicted_cond)
        else:
            st.warning("Condition prediction is unavailable because the ML model is not loaded.")

        st.subheader("🔑 Key Clinical Findings")
        for number, finding in enumerate(key_findings, start=1):
            st.write(f"**{number}.** {finding}")

        st.subheader("📝 AI-Generated Medical Summary")
        if summarizer_loaded:
            st.success(generated_summary)
        else:
            st.error(generated_summary)

        with st.expander("📄 View Original Medical Report"):
            st.write(medical_report)

        with st.expander("🧹 View Preprocessed Medical Report"):
            st.write(cleaned_report)

st.divider()
st.markdown('<div class="custom-footer"><b>Clinical NLP: Automated Medical Report Summarization</b><br><br>Machine Learning • NLP • TF-IDF • Logistic Regression • DistilBART</div>', unsafe_allow_html=True)