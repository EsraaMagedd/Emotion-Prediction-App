

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import re

def load_data(data_path):
    """Loads the data from the CSV file."""
    df = pd.read_csv(data_path)
    df["label"] = df["label"].astype(int)  # Ensure labels are integers
    return df

def preprocess_text(text):
    """Preprocesses text data for model training."""
    text = text.lower()  # Lowercase for case-insensitivity
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
    return text

def train_models(df):
    """Trains the Naive Bayes and Random Forest models."""
    X = df["text"].apply(preprocess_text)
    y = df["label"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Naive Bayes and Random Forest models
    NB_model = MultinomialNB()
    NB_model.fit(X_train_tfidf, y_train)

    RFC_model = RandomForestClassifier()
    RFC_model.fit(X_train_tfidf, y_train)

    return NB_model, RFC_model, vectorizer

def predict_emotion(text, models):
    """Predicts the emotion using the trained models."""
    NB_model, RFC_model, vectorizer = models

    preprocessed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([preprocessed_text])

    NB_prediction = NB_model.predict(text_tfidf)[0]
    RFC_prediction = RFC_model.predict(text_tfidf)[0]

    emotion_predictions = {
        0: ("Sadness", "😢"),
        1: ("Joy", "😃"),
        2: ("Anticipation", "😯"),
        3: ("Anger", "😡"),
        4: ("Fear", "😨"),
        5: ("Surprise", "😲"),
    }

    return NB_prediction, RFC_prediction, emotion_predictions

def main():
    st.set_page_config(page_title="Emotion Prediction", page_icon=":smiley:", layout="wide")

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #B0BAE7; /* Light blue color */
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            text-align: center;
        }
        .stApp > div {
            width: 100%;
        }
        .stTextInput, .stButton {
            margin: 0 auto;
            display: block;
            width: 900px; 
        }
        .textbox {
            width: 400px; /* Set the width of the text box */
            margin: 0 auto;
        }
        .predict-button {
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Emotion Prediction App")
    st.subheader("Describe what you are feeling and see how our models predict its emotion!")

    # Applying the class to the text input element
    user_input = st.text_input("Describe what you are feeling:", key="text_input")
    st.markdown('<div class="textbox">', unsafe_allow_html=True)
    if st.button("Predict"):
        st.markdown('</div>', unsafe_allow_html=True)
        if user_input:
            # Load data and train models (ideally, load pre-trained models)
            data = load_data("data.csv")
            models = train_models(data)  # This could be done outside the main function for efficiency

            # Use st.columns to divide the screen
            col1, col2 = st.columns(2)

            NB_prediction, RFC_prediction, emotion_predictions = predict_emotion(user_input, models)

            with col1:
                st.subheader("Naive Bayes")
                emotion, emoji = emotion_predictions[NB_prediction]
                st.markdown(f"**Predicted Emotion:** {emotion} <br><span style='font-size: 72px;'>{emoji}</span>", unsafe_allow_html=True)

            with col2:
                st.subheader("Random Forest")
                emotion, emoji = emotion_predictions[RFC_prediction]
                st.markdown(f"**Predicted Emotion:** {emotion} <br><span style='font-size: 72px;'>{emoji}</span>", unsafe_allow_html=True)
    else:
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
