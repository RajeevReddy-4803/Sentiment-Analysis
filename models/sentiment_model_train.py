
import os
import pandas as pd
import pickle
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

DATA_PATH = "/kaggle/working/sentiment_analysis/data_handling/customer_feedback_cleaned.csv"
MODEL_SAVE_PATH = "/kaggle/working/sentiment_analysis/models/sentiment_model.pkl"

# Load cleaned data
df = pd.read_csv(DATA_PATH)
df = df[df['sentiment_label'].isin(['positive', 'negative', 'neutral'])]

# Encode labels
label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
id2label = {v: k for k, v in label2id.items()}
df['label'] = df['sentiment_label'].map(label2id)

# Split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['clean_text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

# Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    report_to="none",  
)
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Evaluate
preds = trainer.predict(test_dataset)
pred_labels = preds.predictions.argmax(-1)
print("\n📊 Evaluation Report:")
print(classification_report(test_labels, pred_labels, target_names=label2id.keys()))
print("Accuracy:", accuracy_score(test_labels, pred_labels))

model_save_path = "/kaggle/working/sentiment_analysis/models/sentiment_model.pkl"

with open(model_save_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Sentiment model saved successfully at: {model_save_path}")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n✅ Model saved at: {MODEL_SAVE_PATH}")
