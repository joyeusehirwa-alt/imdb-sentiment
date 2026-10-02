import os
import joblib
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 IMDb Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to predict whether the sentiment is **Positive** or **Negative**.")

# Load model and vectorizer with exact repository filenames
@st.cache_resource
def load_artifacts():
    model = joblib.load("sentiment_model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()

    user_review = st.text_area(
        "Movie Review:",
        placeholder="Type or paste a movie review here...",
        height=150
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not user_review.strip():
            st.warning("Please enter a review first!")
        else:
            transformed_input = vectorizer.transform([user_review])
            prediction = model.predict(transformed_input)[0]

            st.write("---")
            if prediction == 1 or str(prediction).lower() == "positive":
                st.success("### 🎉 Prediction: POSITIVE")
            else:
                st.error("### 😞 Prediction: NEGATIVE")

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(transformed_input)[0]
                st.write(f"**Confidence:** Positive: `{probs[1]:.2%}` | Negative: `{probs[0]:.2%}`")

except Exception as e:
    st.error(f"Error loading model files: {e}")
