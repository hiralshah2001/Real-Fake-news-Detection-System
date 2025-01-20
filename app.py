# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load pre-trained model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('logistic_regression_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    return model, vectorizer

# Streamlit App Title
st.title("Real/Fake News Detection")
st.write("This web app classifies news articles as Real or Fake using a pre-trained Logistic Regression model.")

# Input text area for user input
user_input = st.text_area("Enter news content:", "")

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter news content to classify.")
    else:
        # Load model and vectorizer
        model, vectorizer = load_model()

        # Preprocess input and predict
        user_input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(user_input_tfidf)[0]
        confidence = model.predict_proba(user_input_tfidf)[0]

        # Display results
        if prediction == 1:
            st.error(f"This news is classified as **Fake News** with {confidence[1] * 100:.2f}% confidence.")
        else:
            st.success(f"This news is classified as **Real News** with {confidence[0] * 100:.2f}% confidence.")

# Sidebar Information
st.sidebar.header("About the App")
st.sidebar.info(
    "This app uses a Logistic Regression model trained on TF-IDF vectorized data to classify news articles. It demonstrates the power of NLP and Machine Learning in detecting fake news."
)
