"""
Sentiment Analysis System
Classifies customer reviews as Positive or Negative using TF-IDF
features with Logistic Regression and Naive Bayes.
"""

import pandas as pd
import numpy as np
import re
import string
import joblib
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ===========================================================
# PHASE 1: Setup — imports + NLTK resources
# ===========================================================
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# ===========================================================
# PHASE 2: Load dataset and inspect it
# ===========================================================
DATA_PATH = "data/IMDB Dataset.csv"  # adjust filename if different
df = pd.read_csv(DATA_PATH)

print("Shape of dataset:", df.shape)


# ===========================================================
# PHASE 3: Exploratory Data Analysis (EDA)
# ===========================================================
print("\nClass balance:")
print(df['sentiment'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df)
plt.title("Class Balance: Positive vs Negative")
plt.savefig("eda_class_balance.png")
plt.close()

df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(8, 4))
sns.histplot(df['review_length'], bins=50, kde=True)
plt.title("Review Length Distribution")
plt.xlabel("Number of Words")
plt.savefig("eda_review_length.png")
plt.close()


# ===========================================================
# PHASE 4: Text cleaning function + apply to dataset
# ===========================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)                     # remove HTML tags (e.g. <br />)
    text = re.sub(r'[^a-z\s]', ' ', text)                   # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()                # remove extra whitespace
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in STOPWORDS]
    return ' '.join(tokens)

print("\nCleaning text... (this may take a minute on 50k rows)")
df['cleaned_review'] = df['review'].apply(clean_text)
print("Before:", df['review'].iloc[0][:200])
print("After: ", df['cleaned_review'].iloc[0][:200])


# ===========================================================
# PHASE 5: Encode labels + train/test split
# ===========================================================
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")


# ===========================================================
# PHASE 6: TF-IDF vectorization
# ===========================================================
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ===========================================================
# PHASE 7: Train models (Logistic Regression + Naive Bayes)
# ===========================================================
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

print("Training Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)


# ===========================================================
# PHASE 8: Evaluate models
# ===========================================================
def evaluate(model, name):
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=['negative', 'positive']))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['negative', 'positive'],
                yticklabels=['negative', 'positive'])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_')}.png")
    plt.close()

    return acc

lr_acc = evaluate(lr_model, "Logistic Regression")
nb_acc = evaluate(nb_model, "Naive Bayes")


# ===========================================================
# PHASE 9: Save the best model + vectorizer
# ===========================================================
best_model, best_name = (lr_model, "Logistic Regression") if lr_acc >= nb_acc else (nb_model, "Naive Bayes")
print(f"\nBest model: {best_name} (Accuracy: {max(lr_acc, nb_acc):.4f})")

os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("Saved model/sentiment_model.pkl and model/tfidf_vectorizer.pkl")