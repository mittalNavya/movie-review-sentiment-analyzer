"""
Sentiment Analysis System
Loads the saved model and vectorizer, and tests predictions on new text.
"""

import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# ===========================================================
# PHASE 10: Load saved model + vectorizer, test on new text
# ===========================================================
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in STOPWORDS]
    return ' '.join(tokens)


def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "positive" if prediction == 1 else "negative"


# Quick manual test
sample_reviews = [
    "This movie was absolutely fantastic, I loved every minute of it!",
    "Waste of time. The acting was terrible and the plot made no sense.",
    "It was okay, not great but not bad either."
]

for review in sample_reviews:
    result = predict_sentiment(review)
    print(f"Review: {review}\nPredicted Sentiment: {result}\n")