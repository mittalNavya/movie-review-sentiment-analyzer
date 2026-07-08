"""
Sentiment Analysis System
Flask API that serves sentiment predictions from the saved model.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)
CORS(app)

# ===========================================================
# PHASE 11: Load model once at startup + create /predict endpoint
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


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Sentiment Analysis API is running."})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "review" not in data:
        return jsonify({"error": "Please provide a 'review' field in the JSON body."}), 400

    review_text = data["review"]
    cleaned = clean_text(review_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    sentiment = "positive" if prediction == 1 else "negative"

    return jsonify({
        "review": review_text,
        "predicted_sentiment": sentiment
    })


if __name__ == "__main__":
    app.run(debug=True)
