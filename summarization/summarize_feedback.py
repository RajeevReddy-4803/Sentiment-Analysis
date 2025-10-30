import os
import math
import pandas as pd
from pathlib import Path
from typing import List
import nltk
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

CLEANED_PATH = "/kaggle/working/sentiment_analysis/data_handling/customer_feedback_cleaned.csv"
OUT_DIR = "/kaggle/working/sentiment_analysis/summarization"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "summaries_examples.csv")

# Extractive summarizer

def split_sentences(text: str) -> List[str]:
    return nltk.tokenize.sent_tokenize(text)

def extractive_summary(text: str, short_sentences: int = 1, long_sentences: int = 3):
    sents = split_sentences(text)
    if len(sents) == 0:
        return "", ""
    if len(sents) <= max(short_sentences, long_sentences):
        joined = " ".join(sents)
        return joined, joined

    vectorizer = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2))
    try:
        X = vectorizer.fit_transform(sents)
    except ValueError:
        joined = " ".join(sents)
        return joined, joined

    doc_vec = X.mean(axis=0)
    sims = cosine_similarity(X, doc_vec)
    sims = sims.flatten()
    ranked_idx = sims.argsort()[::-1]

    def topk_ordered(k):
        topk = sorted(ranked_idx[:k]) 
        return " ".join([sents[i] for i in topk])

    short = topk_ordered(short_sentences)
    long = topk_ordered(long_sentences)
    return short, long

# Abstractive summarizer (T5-small)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "t5-small"

def load_t5_model():
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    return tokenizer, model

TOKENIZER, MODEL = None, None
def ensure_model_loaded():
    global TOKENIZER, MODEL
    if TOKENIZER is None or MODEL is None:
        TOKENIZER, MODEL = load_t5_model()

def abstractive_summary(text: str, max_length: int = 50, min_length: int = 12, num_beams: int = 4):
    ensure_model_loaded()
    prefix = "summarize: " + text
    inputs = TOKENIZER.encode(prefix, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)

    out = MODEL.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    summary = TOKENIZER.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return summary


# short long text for summarizer

def safe_trim_for_abstractive(text: str, token_limit: int = 450):
    if len(text) <= 4000:  
        return text
    return text[:4000] 

# Run save outputs

def main(sample_n: int = 200):
    # Load cleaned data
    if not os.path.exists(CLEANED_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_PATH}")
    df = pd.read_csv(CLEANED_PATH)
    df_sample = df.sample(n=min(sample_n, len(df)), random_state=42).reset_index(drop=True)

    outputs = []
    for idx, row in df_sample.iterrows():
        text = str(row["clean_text"])
        ex_short, ex_long = extractive_summary(text, short_sentences=1, long_sentences=3)
        trimmed = safe_trim_for_abstractive(text)
        try:
            abs_short = abstractive_summary(trimmed, max_length=30, min_length=8, num_beams=4)
            abs_long = abstractive_summary(trimmed, max_length=80, min_length=20, num_beams=4)
        except Exception as e:
            abs_short, abs_long = ex_short, ex_long

        outputs.append({
            "id": row.get("id", idx),
            "source": row.get("source", ""),
            "clean_text": text,
            "extractive_short": ex_short,
            "extractive_long": ex_long,
            "abstractive_short": abs_short,
            "abstractive_long": abs_long
        })

    out_df = pd.DataFrame(outputs)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved summaries to {OUT_CSV}")
    out_df.head(50).to_csv(os.path.join(OUT_DIR, "summaries_preview.csv"), index=False)
    print("Saved summaries_preview.csv")
    return out_df

if __name__ == "__main__":
    main(sample_n=200)
