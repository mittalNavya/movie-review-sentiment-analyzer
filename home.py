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
# PHASE 2: Load dataset and inspect 
# ===========================================================
DATA_PATH = "data/IMDB Dataset.csv"  # adjust filename if different
df = pd.read_csv(DATA_PATH)

print("Shape of dataset:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nSample rows:")
print(df.head())