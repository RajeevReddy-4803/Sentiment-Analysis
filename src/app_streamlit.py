import streamlit as st
import pandas as pd
import os
import pickle
from transformers import DistilBertTokenizerFast
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

BASE = "/kaggle/working/sentiment_analysis"
DATA_CLEAN = os.path.join(BASE, "data_handling", "customer_feedback_cleaned.csv")
REPORT_DIR = os.path.join(BASE, "reports")
MODEL_PKL = os.path.join(BASE, "models", "sentiment_model.pkl")
SUM_DIR = os.path.join(BASE, "summarization")

st.set_page_config(page_title="Customer Feedback Analyzer", layout="wide")
st.title("Intelligent Customer Feedback Analysis — Demo")

@st.cache_data
def load_cleaned():
    if os.path.exists(DATA_CLEAN):
        return pd.read_csv(DATA_CLEAN)
    return pd.DataFrame(columns=["id","source","clean_text","sentiment_label"])

df = load_cleaned()

uploaded = st.file_uploader("Upload CSV with columns ['clean_text']", type=['csv'])
if uploaded is not None:
    df_upload = pd.read_csv(uploaded)
    if 'clean_text' in df_upload.columns:
        df = df_upload
        st.success("Uploaded dataset loaded.")
    else:
        st.error("Uploaded CSV must contain 'clean_text' column.")
st.sidebar.header("Options")
show_sample = st.sidebar.checkbox("Show data sample", True)
if show_sample:
    st.subheader("Sample data")
    st.dataframe(df.head(50))

# Sentiment distribution
st.subheader("Sentiment distribution (precomputed)")
if 'sentiment_label' in df.columns:
    dist = df['sentiment_label'].value_counts()
    st.bar_chart(dist)
else:
    st.info("No sentiment_label found - run preprocessing first.")

# Load topics and forecast images if exist
st.subheader("Top recurring issues (topics)")
topics_csv = os.path.join(REPORT_DIR, "topics.csv")
if os.path.exists(topics_csv):
    topics = pd.read_csv(topics_csv)
    st.table(topics.head(8))
else:
    st.info("No topics.csv found. Run insights script to generate topics.")

st.subheader("Forecast (next 30 days)")
forecast_png = os.path.join(REPORT_DIR, "forecast.png")
if os.path.exists(forecast_png):
    st.image(forecast_png, use_column_width=True)
else:
    st.info("No forecast plot found. Run insights script.")

# Load model for prediction
def load_model_pickle():
    if os.path.exists(MODEL_PKL):
        try:
            with open(MODEL_PKL, "rb") as f:
                model = pickle.load(f)
            tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            return model, tokenizer
        except Exception as e:
            st.error("Failed to load model pickle: " + str(e))
    return None, None

st.subheader("Quick text prediction")
model, tokenizer = load_model_pickle()
txt = st.text_area("Enter feedback text to classify", height=120)
if st.button("Predict"):
    if model is None:
        st.error("Model not found. Ensure models/sentiment_model.pkl exists.")
    else:
        inputs = tokenizer(txt, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            out = model(**inputs)
            pred = torch.argmax(out.logits, dim=1).item()
            label_map = {0:"negative",1:"neutral",2:"positive"}
            st.success(f"Prediction: {label_map.get(pred,'unknown')}")

# Summaries preview 
st.subheader("Summaries preview")
summ_preview = os.path.join(SUM_DIR, "summaries_preview.csv")
if os.path.exists(summ_preview):
    pp = pd.read_csv(summ_preview)
    st.dataframe(pp[['id','source','clean_text','extractive_short','abstractive_short']].head(20))
else:
    st.info("No summaries_preview.csv found - run summarization script.")

st.sidebar.markdown("---")
st.sidebar.markdown("Made for assignment submission")
