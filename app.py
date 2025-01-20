# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load pre-trained model and vectorizer
@st.cache
def load_model():
    return joblib.load('logistic_model.pkl'), joblib.load('tfidf_vectorizer.pkl')

# Title and description
st.title("Real/Fake News Detection")
st.write("This is a web application that classifies news articles as Real or Fake.")

# Input form
user_input = st.text_area("Enter the news content here:", "")

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter news content for classification.")
    else:
        # Load model and vectorizer
        model, vectorizer = load_model()

        # Preprocess and predict
        user_input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(user_input_tfidf)[0]
        confidence = model.predict_proba(user_input_tfidf)[0]

        # Display results
        if prediction == 1:
            st.error(f"This news is classified as **Fake News** with {confidence[1]*100:.2f}% confidence.")
        else:
            st.success(f"This news is classified as **Real News** with {confidence[0]*100:.2f}% confidence.")

# Additional UI
st.sidebar.header("About the App")
st.sidebar.info("This app uses a pre-trained Logistic Regression model and TF-IDF vectorization for classifying news articles.")
