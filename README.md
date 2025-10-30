# Sentiment Analysis

This repository contains the end-to-end pipeline for sentiment analysis including data cleaning, modeling, summarization, forecasting, and an interactive app.

## 📁 Directory Structure

```
data_handling/    # Data cleaning/preprocessing scripts and cleaned CSVs
models/           # Sentiment model training scripts and trained model artifacts
summarization/    # Feedback summarizer code and outputs
src/              # Insights forecast and Streamlit app code
reports/          # Notebooks and visual/report outputs
raw_data/         # Original data (not included in repo due to size)
```

## ⚙️ Setup & Dependencies

Clone this repository, then create and activate a virtual environment:

```bash
python -m venv venv  # Windows: venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt  # Or see below for all needed packages
```

Or manually install:
```
pip install pandas numpy matplotlib seaborn scikit-learn torch torchvision torchaudio transformers datasets sentencepiece tqdm nltk wordcloud textblob plotly streamlit prophet
```

## ▶️ Pipeline Usage Order

1. **Preprocessing**
    - `python data_handling/data_preprocessing.py`
2. **Sentiment Model**
    - `python models/sentiment_model_train.py`
3. **Summarization**
    - `python summarization/summarize_feedback.py`
4. **Insights Notebook**
    - Open & run `reports/insights.ipynb` (run all cells)
5. **Forecast & App**
    - `python src/insights_forecast.py`
    - `streamlit run src/app_streamlit.py`

## Notes
- Only key code scripts and notebook are tracked in git (`.gitignore` masks large/intermediate files and outputs).
- For full outputs or data, re-run above scripts on your setup.
- For more, see [Sentiment-Analysis on GitHub](https://github.com/RajeevReddy-4803/Sentiment-Analysis).
