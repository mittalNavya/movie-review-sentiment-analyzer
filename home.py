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
print("\nColumn names:", df.columns.tolist())
print("\nSample rows:")
print(df.head())


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

print("\nEDA plots saved: eda_class_balance.png, eda_review_length.png")