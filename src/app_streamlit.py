
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import textwrap
from pathlib import Path
from io import StringIO
import matplotlib.pyplot as plt
import plotly.express as px
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chatbot
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
    T5_AVAILABLE = True
except Exception:
    T5_AVAILABLE = False

# =============== CONFIG PATHS ===============
BASE = Path.cwd()
DATA_PATH = BASE / "data_handling" / "customer_feedback_cleaned.csv"
MODEL_PATH = BASE / "models" / "sentiment_model.pkl"
SUM_PREVIEW = BASE / "summarization" / "summaries_preview.csv"
TOPICS_CSV = BASE / "reports" / "topics.csv"
FORECAST_IMG = BASE / "reports" / "forecast.png"

# =============== PAGE CONFIG ===============
st.set_page_config(page_title="Customer Feedback Analyzer", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    body, .stApp, .reportview-container {
        background: linear-gradient(120deg, #232526 60%, #20232c 100%);
        color: #e4e6f6;
    }
    h1, h2, h3, h4 { color: #29b6f6; font-family: 'Segoe UI', Verdana, Geneva, sans-serif; }
    .st-bj { background-color: #232526; }
    .stButton>button {
        background: linear-gradient(90deg, #29b6f6, #1565c0);
        color: #fff; font-weight: 600; border-radius:8px; min-width:130px;
        box-shadow: 0 2px 12px #1c2833a0; padding: 8px 18px;
    }
    .metric-box { background:#1a1a23; border-radius:10px; padding:20px; margin:10px 0; }
    .stDataFrame {background:#191934}
    .st-cw { font-size: 1.2rem; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_cleaned(path):
    if path.exists():
        df = pd.read_csv(path)
        return df
    return pd.DataFrame()

def load_hf_model(model_dir):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load Huggingface model: {e}")
        return None, None

if (BASE / "models" / "sentiment_model").exists() and (BASE / "models" / "sentiment_model").is_dir():
    model, tokenizer = load_hf_model(BASE / "models" / "sentiment_model")
else:
    model, tokenizer = None, None
    if MODEL_PATH.exists() and MODEL_PATH.suffix==".pkl":
        st.warning("It looks like you have a .pkl model but this app expects Huggingface 'save_pretrained' transformer models inside a folder. Please export and save your transformer model using 'save_pretrained'.")

def predict_text(model, tokenizer, text):
    if model is None or tokenizer is None or not isinstance(text, str) or text.strip()=="":
        return "model_missing"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits if hasattr(out, "logits") else out[0]
        pred = int(torch.argmax(logits, dim=1).item())
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return label_map.get(pred, str(pred))

def extractive_summary(text, short_sentences=1, long_sentences=3):
    import nltk
    nltk.download('punkt', quiet=True)
    sents = nltk.tokenize.sent_tokenize(text)
    if len(sents) == 0:
        return "", ""
    if len(sents) <= max(short_sentences, long_sentences):
        joined = " ".join(sents)
        return joined, joined
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2), stop_words='english')
    try:
        X = vectorizer.fit_transform(sents)
    except Exception:
        joined = " ".join(sents)
        return joined, joined
    doc_vec = X.mean(axis=0)
    sims = cosine_similarity(X, doc_vec).flatten()
    ranked_idx = sims.argsort()[::-1]
    def topk(k):
        idxs = sorted(ranked_idx[:k])
        return " ".join([sents[i] for i in idxs])
    return topk(short_sentences), topk(long_sentences)

@st.cache_resource
def load_t5():
    if not T5_AVAILABLE:
        return None, None
    try:
        tokenizer = T5TokenizerFast.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small").to("cuda" if torch.cuda.is_available() else "cpu")
        return tokenizer, model
    except Exception:
        return None, None

def abstractive_summary(t5_tok, t5_model, text, max_len=50, min_len=10):
    if t5_tok is None or t5_model is None:
        return ""
    prefix = "summarize: " + text
    inputs = t5_tok.encode(prefix, return_tensors="pt", max_length=512, truncation=True).to(next(t5_model.parameters()).device)
    outs = t5_model.generate(inputs, max_length=max_len, min_length=min_len, num_beams=4, early_stopping=True, no_repeat_ngram_size=2)
    return t5_tok.decode(outs[0], skip_special_tokens=True)

st.sidebar.title("Options")
show_sample = st.sidebar.checkbox("Show data sample", value=True)
st.sidebar.markdown("Made for assignment — quick demo")

# App title
st.title("Intelligent Customer Feedback Analysis System using AI")
st.markdown(
    """
    <div style='color:#f6f9fa;font-size:18px;margin-bottom:1rem;'>
        Unlock rich insights, diagnose trends, and interact with AI about your customer feedback—all in one sleek dashboard.
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([4,1])
with col1:
    uploaded = st.file_uploader("⬆️ Upload CSV (with 'clean_text' column)", type=["csv"])
    if uploaded:
        try:
            uploaded_df = pd.read_csv(uploaded)
            st.success("Uploaded CSV loaded.")
            cleaned_df = uploaded_df.copy()
        except Exception as e:
            st.error(f"❌ Failed to read uploaded CSV: {e}")
            cleaned_df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()
    else:
        cleaned_df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()
        if cleaned_df.empty:
            st.warning("No cleaned dataset found. Upload or preprocess data to continue.")
with col2:
    st.markdown("**Rows available:**")
    st.metric("Records", int(len(cleaned_df)) if not cleaned_df.empty else 0)

if not cleaned_df.empty and st.checkbox("Show data sample", value=True, key="sample_data_chkbox"):
    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    st.subheader("🔎 Data sample:")
    st.dataframe(cleaned_df.head(25))
    st.markdown("</div>", unsafe_allow_html=True)
elif cleaned_df.empty:
    st.info("No data to display. Upload or run preprocessing.")

if 'sentiment_label' in cleaned_df.columns and not cleaned_df.empty:
    st.subheader("Sentiment distribution")
    dist = cleaned_df['sentiment_label'].value_counts().reset_index()
    dist.columns = ['label','count']
    fig = px.pie(dist, names='label', values='count',
        title='', color_discrete_sequence=px.colors.sequential.Teal)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No sentiment_label column in data. Run model or preprocessing to analyze sentiment.")

left, right = st.columns([2,1])
with left:
    st.subheader("Top recurring issues")
    if TOPICS_CSV.exists():
        try:
            st.table(pd.read_csv(TOPICS_CSV).head(8))
        except Exception as e:
            st.warning(f"Can't load topics.csv: {e}")
    else:
        st.info("No topic breakdown found. Run insights_forecast.py to generate.")
with right:
    st.subheader("Forecast (30 days)")
    if FORECAST_IMG.exists():
        st.image(str(FORECAST_IMG), use_column_width=True)
    else:
        st.info("No forecast found. Run insights_forecast.py to generate.")

st.markdown("---")

st.markdown("---")

if SUM_PREVIEW.exists():
    s_preview = pd.read_csv(SUM_PREVIEW)
    st.subheader("Preview: auto-generated summaries")
    st.dataframe(s_preview[['id','source','clean_text','extractive_short','abstractive_short']].head(30), height=330)
else:
    st.info("No summaries_preview.csv found - run summarization script first.")

st.markdown("---")

st.subheader("🤖 Advanced Feedback Chatbot (GPT-2)")
chat_query = st.text_input("Ask anything about feedback trends, issues, or quality:", key="chat_q")
if st.button("Chat with AI", key="chat_ai"):
    context = ""
    if SUM_PREVIEW.exists():
        try:
            sp = pd.read_csv(SUM_PREVIEW)
            context = " ".join(sp['abstractive_short'].dropna().astype(str).head(7).tolist())
        except Exception:
            context = ""
    reply = chatbot.generate_response(chat_query, context)
    st.markdown(f"**AI:** {reply}")

st.markdown("---")
with st.container():
    st.markdown("### Download Outputs:")
    c1, c2, c3 = st.columns(3)
    if DATA_PATH.exists():
        c1.download_button("Download Cleaned CSV", data=DATA_PATH.read_bytes(), file_name="customer_feedback_cleaned.csv")
    if MODEL_PATH.exists():
        c2.download_button("Download Model (pkl)", data=MODEL_PATH.read_bytes(), file_name="sentiment_model.pkl")
    if SUM_PREVIEW.exists():
        c3.download_button("Download Summaries", data=SUM_PREVIEW.read_bytes(), file_name="summaries_preview.csv")

st.caption("Professional AI Dashboard — for advanced customer feedback review. For errors, check preprocessing and model scripts.")
