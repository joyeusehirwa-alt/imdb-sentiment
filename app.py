import os
import joblib
import streamlit as st

# Set page title and layout
st.set_page_config(
    page_title="IMDb Sentiment Analyzer", page_icon="🎬", layout="centered"
)

st.title("🎬 IMDb Movie Review Sentiment Analyzer")
st.write(
    "Enter a movie review below to predict whether the sentiment is **Positive** or **Negative**."
)


# Load model and vectorizer
@st.cache_resource
def load_artifacts():
    # Make sure filenames match what is saved in your repository
    model_path = "sentiment_model.joblib"
    vectorizer_path = "tfidf_vectorizer.joblib"

    if not os.path.exists(model_path):
        # Fallback in case vectorizer/model filenames differ slightly
        model_path = "model.joblib"
    if not os.path.exists(vectorizer_path):
        vectorizer_path = "vectorizer.joblib"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


try:
    model, vectorizer = load_artifacts()

    # User input text area
    user_review = st.text_area(
        "Movie Review:",
        placeholder="Type or paste a movie review here...",
        height=150,
    )

    if st.button("Analyze Sentiment", type="primary"):
        if user_review.strip() == "":
            st.warning("Please enter a review first!")
        else:
            # Transform and predict
            transformed_input = vectorizer.transform([user_review])
            prediction = model.predict(transformed_input)[0]

            # Display results
            st.write("---")
            if prediction == 1 or prediction == "positive":
                st.success("### 🎉 Prediction: POSITIVE")
            else:
                st.error("### 😞 Prediction: NEGATIVE")

            # Show prediction probability if supported by model
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(transformed_input)[0]
                st.write(
                    f"**Confidence:** Positive: `{probs[1]:.2%}` | Negative: `{probs[0]:.2%}`"
                )

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info(
        "Make sure your `.joblib` model and vectorizer files are uploaded to the main folder of your repository."
    )
