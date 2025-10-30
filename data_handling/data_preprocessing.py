import os
import re
import pandas as pd
import numpy as np
import nltk
import string
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# === Paths ===
RAW_DIR = "/kaggle/input/raw-data-sentiment"
SAVE_DIR = "/kaggle/working/sentiment_analysis/data_handling"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load datasets ===
files = os.listdir(RAW_DIR)
print("Available raw files:", files)

twitter_path = os.path.join(RAW_DIR, "Tweets.csv")
yelp_train = os.path.join(RAW_DIR, "train.csv")
yelp_test = os.path.join(RAW_DIR, "test.csv")

df_twitter = pd.read_csv(twitter_path)
df_yelp_train = pd.read_csv(yelp_train)
df_yelp_test = pd.read_csv(yelp_test)
df_yelp = pd.concat([df_yelp_train, df_yelp_test], ignore_index=True)

df_twitter["source"] = "twitter"
df_yelp["source"] = "yelp"

print(f"Twitter shape: {df_twitter.shape}")
print(f"Yelp shape: {df_yelp.shape}")
print("\nTwitter columns:", df_twitter.columns.tolist())
print("Yelp columns:", df_yelp.columns.tolist())

# === Standardize schema ===
def standardize(df, text_col, label_col, source_name):
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df[["text", "label"]].copy()
    df["source"] = source_name
    return df

twitter_text = [c for c in df_twitter.columns if "text" in c.lower()][0]
twitter_label = [c for c in df_twitter.columns if "sentiment" in c.lower() or "airline_sentiment" in c.lower()][0]
yelp_text = [c for c in df_yelp.columns if "text" in c.lower() or "review" in c.lower()][0]
yelp_label = [c for c in df_yelp.columns if "sentiment" in c.lower() or "label" in c.lower() or "stars" in c.lower()][0]

df_twitter = standardize(df_twitter, twitter_text, twitter_label, "twitter")
df_yelp = standardize(df_yelp, yelp_text, yelp_label, "yelp")

# === Merge + clean ===
df = pd.concat([df_twitter, df_yelp], ignore_index=True)
df.dropna(subset=["text"], inplace=True)
df["id"] = np.arange(len(df))

nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

tqdm.pandas()
df["clean_text"] = df["text"].progress_apply(clean_text)

def map_label(label):
    label = str(label).lower()
    if any(x in label for x in ["pos", "5", "4", "good", "happy", "great"]):
        return "positive"
    elif any(x in label for x in ["neg", "1", "2", "bad", "poor", "angry"]):
        return "negative"
    else:
        return "neutral"

df["sentiment_label"] = df["label"].apply(map_label)
df.drop_duplicates(subset=["clean_text"], inplace=True)
df = df[df["clean_text"].str.len() > 3]

final_df = df[["id", "source", "clean_text", "sentiment_label"]].reset_index(drop=True)
save_path = os.path.join(SAVE_DIR, "customer_feedback_cleaned.csv")
final_df.to_csv(save_path, index=False)

print(f"\n✅ Cleaned dataset saved at: {save_path}")
print(f"Total cleaned records: {len(final_df)}")

# === Visualization ===
final_df['sentiment_label'].value_counts().plot(kind='bar', title='Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "sentiment_distribution.png"))
plt.close()

print(final_df.groupby('source')['sentiment_label'].value_counts(normalize=True))
