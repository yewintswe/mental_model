import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Mental Health Screening",
    page_icon="🌸",
    layout="centered",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #ffffff !important;
    color: #1a1a1a;
}
[data-testid="stHeader"] { background: transparent; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #fce4ec !important;
    border-right: 2px solid #f48fb1;
}
[data-testid="stSidebar"] * { color: #1a1a1a !important; }
.sidebar-title {
    font-size: 13px; font-weight: 700;
    color: #880e4f !important; margin-bottom: 2px;
}
.sidebar-val {
    font-size: 12px; color: #444 !important; margin-bottom: 6px;
}
.sidebar-divider {
    border-top: 1px solid #f48fb1; margin: 12px 0;
}

/* Page header */
.page-header {
    background: linear-gradient(135deg, #f8bbd0 0%, #fce4ec 100%);
    border-radius: 12px;
    padding: 24px 28px 18px;
    margin-bottom: 26px;
    border: 1px solid #f48fb1;
}
.page-header h1 { font-size: 24px; font-weight: 700; color: #880e4f; margin: 0 0 6px; }
.page-header p  { font-size: 13px; color: #555; margin: 0; }

/* Section headers */
.sec-head {
    font-size: 13px; font-weight: 700; color: #880e4f;
    text-transform: uppercase; letter-spacing: .5px;
    border-bottom: 2px solid #f8bbd0; padding-bottom: 4px;
    margin: 20px 0 10px;
}

/* Widget labels */
label, div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #1a1a1a !important;
}

/* Predict button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #e91e63, #f06292) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 10px 0 !important;
    width: 100% !important;
    margin-top: 10px;
}
div[data-testid="stButton"] > button:hover { opacity: 0.87; }

/* Result cards */
.r-card {
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.r-card.yes { background: #fce4ec; border: 2px solid #e91e63; }
.r-card.no  { background: #f1f8e9; border: 2px solid #8bc34a; }
.r-condition { font-size: 13px; font-weight: 600; color: #555; margin-bottom: 2px; }
.r-label    { font-size: 22px; font-weight: 800; }
.r-label.yes { color: #c2185b; }
.r-label.no  { color: #558b2f; }
.r-prob { font-size: 12px; color: #555; margin-top: 4px; }
.bar-bg { background:#e0e0e0; border-radius:6px; height:8px; margin-top:6px; }
.bar-fg { height:8px; border-radius:6px; }

/* Disclaimer */
.disclaimer {
    background: #fff8e1; border-left: 4px solid #ffb300;
    border-radius: 6px; padding: 12px 16px;
    font-size: 12px; color: #555; margin-top: 18px;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    with open("models.pkl", "rb") as f:
        return pickle.load(f)

try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("⚠️  models.pkl not found. Run `python model_training.py` first.")
    st.stop()

anxiety_model    = bundle["anxiety_model"]
depression_model = bundle["depression_model"]
ptsd_model       = bundle["ptsd_model"]
label_encoders   = bundle["label_encoders"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Replace the line below with your actual university logo ──
    # st.image("university_logo.png", use_container_width=True)
    st.image(r"D:\lumi\images.jpg", use_container_width=True)    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-title">Student Name</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-val">Ye Wint Swe</p>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-title">Student ID</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-val">PIUS20220029</p>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-title">Supervisor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-val">Dr. Nwe Nwe Htay Win</p>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-val">Capstone Project 2025/2026</p>', unsafe_allow_html=True)

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h1>🌸 Student Mental Health Screening</h1>
  <p>Fill in the information below and click <strong>Predict</strong> to receive a screening result
     for anxiety, depression, and PTSD.</p>
</div>
""", unsafe_allow_html=True)

# ── Helper ─────────────────────────────────────────────────────────────────────
def encode(col, value):
    return int(label_encoders[col].transform([value])[0])

# ── Form ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Demographics</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
age            = c1.number_input("Age", min_value=10, max_value=100, value=22, step=1)
gender         = c2.selectbox("Gender", ["Female", "Male"])
marital_status = c3.selectbox("Marital Status", ["Single", "Married", "Divorced", "Other"])

c4, c5 = st.columns(2)
education   = c4.selectbox("Education Level", ["High school", "Higher education"])
employment  = c5.selectbox("Employment Status", ["Student", "Employed", "Unemployed"])

c6, c7 = st.columns(2)
region      = c6.selectbox("Region", [
    "Yangon Region", "Mandalay Region", "Ayeyarwady Region", "Bago Region",
    "Chin State", "Kachin State", "Kayah State ( Karenni)", "Kayin State",
    "Magway Region", "Mon State", "Rakhine State", "Sagaing Region",
    "Shan State", "Tanintharyi Region"
])
living_area = c7.selectbox("Living Area", [
    "Urban ( City, Town)", "Rural (villages, countryside)"
])

# ── Financial ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Financial Situation</div>', unsafe_allow_html=True)

f1, f2, f3 = st.columns(3)
income_group           = f1.selectbox("Income Level",        ["High income", "Middle income", "Low income"])
financial_struggle     = f2.selectbox("Financial Struggle",  ["Low", "Moderate", "High"])
bill_difficulty        = f3.selectbox("Bill Difficulty",     ["Low", "Moderate", "High"])

f4, f5 = st.columns(2)
income_stability       = f4.selectbox("Income Stability",    ["Stable", "Unstable", "No income"])
financial_stress       = f5.selectbox("Financial Stress",    ["Low", "Moderate", "High"])

# ── Living conditions ──────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Living Conditions</div>', unsafe_allow_html=True)

l1, l2, l3 = st.columns(3)
healthcare_access  = l1.selectbox("Healthcare Access",   ["Has access", "Limited/no access"])
housing            = l2.selectbox("Housing Situation",   ["Stable housing", "Unstable housing"])
conflict_impact    = l3.selectbox("Conflict Impact",     ["No/Mild Impact", "Moderate/Severe Impact", "Other"])
service_access     = st.selectbox("Service Access",      ["Easy", "Moderate", "Hard"])

# ── Predict ────────────────────────────────────────────────────────────────────
predict_btn = st.button("🔍  Predict")

if predict_btn:
    input_data = {
        "age":                      age,
        "marital_status":           encode("marital_status",           marital_status),
        "region":                   encode("region",                   region),
        "living_area":              encode("living_area",              living_area),
        "gender_group":             encode("gender_group",             gender),
        "education_group":          encode("education_group",          education),
        "employment_group":         encode("employment_group",         employment),
        "income_group":             encode("income_group",             income_group),
        "financial_struggle_group": encode("financial_struggle_group", financial_struggle),
        "bill_difficulty_group":    encode("bill_difficulty_group",    bill_difficulty),
        "income_stability_group":   encode("income_stability_group",   income_stability),
        "financial_stress_group":   encode("financial_stress_group",   financial_stress),
        "healthcare_access_group":  encode("healthcare_access_group",  healthcare_access),
        "housing_group":            encode("housing_group",            housing),
        "conflict_impact_group":    encode("conflict_impact_group",    conflict_impact),
        "service_access_group":     encode("service_access_group",     service_access),
    }

    X_input = pd.DataFrame([input_data])

    def get_result(model, X):
        pred  = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        pos_idx = classes.index(1) if 1 in classes else 0
        return int(pred), float(proba[pos_idx])

    anx_pred,  anx_prob  = get_result(anxiety_model,    X_input)
    dep_pred,  dep_prob  = get_result(depression_model, X_input)
    ptsd_pred, ptsd_prob = get_result(ptsd_model,       X_input)

    # Map encoded 0/1 back to readable labels
    anx_le   = label_encoders["anxiety_binary"]
    dep_le   = label_encoders["depression_binary"]
    ptsd_le  = label_encoders["ptsd_binary"]

    anx_label  = anx_le.inverse_transform([anx_pred])[0]
    dep_label  = dep_le.inverse_transform([dep_pred])[0]
    ptsd_label = ptsd_le.inverse_transform([ptsd_pred])[0]

    def is_positive(label):
        """Return True if the prediction indicates symptoms present."""
        return "symptom" in label.lower() or label.lower() == "yes"

    results = [
        ("Anxiety",    anx_label,  anx_prob,  is_positive(anx_label)),
        ("Depression", dep_label,  dep_prob,  is_positive(dep_label)),
        ("PTSD",       ptsd_label, ptsd_prob, is_positive(ptsd_label)),
    ]

    st.markdown("---")
    st.markdown("### 📊 Screening Results")

    for condition, label, prob, positive in results:
        css   = "yes" if positive else "no"
        emoji = "⚠️" if positive else "✅"
        color = "#e91e63" if positive else "#8bc34a"
        pct   = round(prob * 100, 1)

        st.markdown(f"""
        <div class="r-card {css}">
          <div class="r-condition">{emoji} {condition}</div>
          <div class="r-label {css}">{label}</div>
          <div class="r-prob">Probability: <strong>{pct}%</strong></div>
          <div class="bar-bg">
            <div class="bar-fg" style="width:{pct}%; background:{color};"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
      ⚠️ <strong>Disclaimer:</strong> This tool is for academic research purposes only and does
      not constitute a clinical diagnosis. Please consult a qualified mental health professional
      for any concerns about your mental health.
    </div>
    """, unsafe_allow_html=True)
