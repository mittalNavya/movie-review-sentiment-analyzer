# CineSense

An end-to-end NLP system that classifies movie reviews as **positive** or **negative**, built with TF-IDF + Logistic Regression and deployed as a live Flask REST API.

**Live Demo API:** https://movie-review-sentiment-analyzer-8gf3.onrender.com

---

## Overview

This project takes raw, unstructured movie reviews and predicts their sentiment. It covers the full machine learning lifecycle: data cleaning, feature engineering, model training and evaluation, API development, and cloud deployment.

## Dataset

- **Source:** IMDB Dataset of 50K Movie Reviews (Kaggle)
- **Size:** 50,000 reviews, evenly balanced (25,000 positive / 25,000 negative)

## Approach

1. **Text Cleaning** — lowercasing, HTML tag removal, punctuation removal, stopword removal, lemmatization
2. **Feature Engineering** — TF-IDF vectorization (unigrams + bigrams, top 5,000 features)
3. **Modeling** — trained and compared two models:
   - Logistic Regression — **89.19% accuracy** (selected as final model)
   - Multinomial Naive Bayes — 86.12% accuracy
4. **Evaluation** — accuracy, precision, recall, F1-score, and confusion matrix
5. **Deployment** — served via a Flask REST API, deployed on Render, with a simple HTML frontend for live demos

## Project Structure

```
sentiment-analysis/
├── home.py              # Data loading, cleaning, training, evaluation
├── predict.py           # Standalone script to test predictions
├── app.py                # Flask API (/predict endpoint)
├── index.html            # Browser-based demo frontend
├── requirements.txt      # Python dependencies
├── model/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
└── data/                 # Dataset (not tracked in git)
```

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (generates model/ files)
python home.py

# Test predictions locally
python predict.py

# Run the API
python app.py
```

## API Usage

**Endpoint:** `POST /predict`

**Request body:**
```json
{
  "review": "This movie was absolutely wonderful!"
}
```

**Response:**
```json
{
  "review": "This movie was absolutely wonderful!",
  "predicted_sentiment": "positive"
}
```

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 89.19% |
| Naive Bayes | 86.12% |

## Limitations

- The model relies on word-level patterns and does not understand sarcasm, tone, or context — sarcastic reviews may be misclassified.
- Trained specifically on movie reviews; performance may vary on other domains (e.g. product reviews).

## Future Enhancements

- Incorporate context-aware deep learning models (e.g. BERT) to better handle sarcasm and nuanced language
- Expand to multi-class sentiment (positive / negative / neutral)
- Add confidence scores to predictions

## Tech Stack

Python, scikit-learn, NLTK, Flask, Render (deployment)
