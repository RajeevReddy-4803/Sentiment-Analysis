
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from prophet import Prophet
from datetime import timedelta
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# Paths
BASE_DIR = "/kaggle/working/sentiment_analysis"
DATA_PATH = os.path.join(BASE_DIR, "data_handling", "customer_feedback_cleaned.csv")
OUT_DIR = os.path.join(BASE_DIR, "reports")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA_PATH)
print("Loaded cleaned data:", len(df))

score_map = {"positive": 5.0, "neutral": 3.0, "negative": 1.0}
df['score'] = df['sentiment_label'].map(score_map).astype(float)

if 'datetime' in df.columns:
    df['date'] = pd.to_datetime(df['datetime']).dt.date
else:
    n = len(df)
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=180)
    dates = [start + pd.Timedelta(days=int(i*180/n)) for i in range(n)]
    df['date'] = pd.to_datetime(dates).date

daily = df.groupby('date')['score'].mean().reset_index().rename(columns={'date':'ds','score':'y'})
daily['ds'] = pd.to_datetime(daily['ds'])
daily = daily.sort_values('ds')
daily.to_csv(os.path.join(OUT_DIR, "daily_scores.csv"), index=False)


# Forecast using Prophet

model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
model.fit(daily)

future = model.make_future_dataframe(periods=30)  # next 30 days
forecast = model.predict(future)

# Save forecast plot
fig = model.plot(forecast, xlabel='Date', ylabel='Avg Satisfaction Score')
plt.title('Customer Satisfaction Forecast (next 30 days)')
fig_path = os.path.join(OUT_DIR, "forecast.png")
fig.savefig(fig_path, bbox_inches='tight')
plt.close()

forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv(os.path.join(OUT_DIR, "forecast.csv"), index=False)

# Topic modeling

texts = df['clean_text'].fillna("").astype(str).tolist()
tfidf = TfidfVectorizer(max_df=0.9, min_df=5, max_features=5000, ngram_range=(1,2), stop_words='english')
X = tfidf.fit_transform(texts)

n_topics = 8
nmf = NMF(n_components=n_topics, random_state=42, init='nndsvda', max_iter=200)
W = nmf.fit_transform(X)
H = nmf.components_
feature_names = tfidf.get_feature_names_out()

topn = 10
topics = []
for t in range(n_topics):
    top_idx = H[t].argsort()[-topn:][::-1]
    top_words = [feature_names[i] for i in top_idx]
    topics.append((t, top_words))

topics_df = pd.DataFrame({
    "topic": [t for t, w in topics],
    "top_keywords": ["; ".join(w) for t,w in topics]
})
topics_df.to_csv(os.path.join(OUT_DIR, "topics.csv"), index=False)

# Save a bar chart of top keywords 
for t in range(min(4, n_topics)):
    words = topics[t][1][:10]
    values = [H[t, feature_names.tolist().index(w)] if w in feature_names else 0 for w in words]
    plt.figure(figsize=(8,4))
    plt.barh(words[::-1], values[::-1])
    plt.title(f"Topic {t}: top keywords")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"topic_{t}_keywords.png"))
    plt.close()

# Generate a one-page PDF report with visuals & insights
pdf_path = os.path.join(OUT_DIR, "AI_insights_report.pdf")
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()
flow = []

flow.append(Paragraph("AI Insights Report", styles['Title']))
flow.append(Spacer(1,12))
flow.append(Paragraph("Summary", styles['Heading2']))
flow.append(Paragraph(
    "This report summarizes recurring issues discovered via topic modeling and provides a 30-day forecast of average customer satisfaction "
    "scores derived from sentiment labels.", styles['BodyText']))
flow.append(Spacer(1,12))

# Forecast image
flow.append(Paragraph("Forecast of Average Satisfaction Score (next 30 days)", styles['Heading3']))
flow.append(Image(fig_path, width=480, height=240))
flow.append(Spacer(1,12))

# Topics summary
flow.append(Paragraph("Top recurring issues (topics)", styles['Heading3']))
for t,w in topics:
    flow.append(Paragraph(f"Topic {t}: " + ", ".join(w[:8]), styles['BodyText']))
flow.append(Spacer(1,12))

flow.append(Paragraph("Notes & Recommendations", styles['Heading3']))
flow.append(Paragraph(
    "- Topic modeling shows recurring themes; investigate top topics with highest volume.\n"
    "- Forecast indicates expected change in average satisfaction; if yhat trends downward, prioritize high-impact fixes.\n"
    "- Consider monitoring topic frequency over time (weekly) to measure remediation impact.",
    styles['BodyText']
))
doc.build(flow)

print("Saved outputs to:", OUT_DIR)
print(" - daily_scores.csv")
print(" - forecast.csv / forecast.png")
print(" - topics.csv and topic_*_keywords.png")
print(" - AI_insights_report.pdf")
