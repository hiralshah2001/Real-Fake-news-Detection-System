import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("logistic_regression_model.joblib")  # Your saved model
    vectorizer = joblib.load("tfidf_vectorizer.joblib")  # Your saved TF-IDF vectorizer
    return model, vectorizer

# Streamlit App
st.title("Real/Fake News Detection")
st.write("Classify news articles as Real or Fake using a pre-trained Logistic Regression model.")

# Text area for user input
user_input = st.text_area("Enter news content here:", "")

# Classification button
if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter some text to classify.")
    else:
        # Load model and vectorizer
        model, vectorizer = load_model()

        # Transform user input using the TF-IDF vectorizer
        user_input_tfidf = vectorizer.transform([user_input])

        # Predict and get confidence scores
        prediction = model.predict(user_input_tfidf)[0]
        confidence = model.predict_proba(user_input_tfidf)[0]

        # Display results
        if prediction == 1:
            st.error(f"This news is classified as **Fake News** with {confidence[1] * 100:.2f}% confidence.")
        else:
            st.success(f"This news is classified as **Real News** with {confidence[0] * 100:.2f}% confidence.")

# Sidebar information
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a Logistic Regression model trained with TF-IDF vectorized data. "
    "It is designed to demonstrate the application of machine learning for detecting fake news."
)
