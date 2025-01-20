# app.py
import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("logistic_regression_model.joblib")  # Your saved model
        return model
    except FileNotFoundError as e:
        st.error("Required file 'logistic_regression_model.joblib' not found. Please ensure it is in the correct directory.")
        st.stop()

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
        # Load model
        model = load_model()

        # Transform user input using inline TF-IDF vectorization
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=5000)
        user_input_tfidf = tfidf_vectorizer.fit_transform([user_input])

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
    "It demonstrates the application of machine learning for detecting fake news."
)
