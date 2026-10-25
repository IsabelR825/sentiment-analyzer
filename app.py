import streamlit as st
from transformers import pipeline

st.title("AI Sentiment Analyzer")

@st.cache_resource
def load_model():
    # Load the sentiment analysis model from Hugging Face
    return pipeline("text-classification",
        model="j-hartmann/emotion-english-distilroberta-base")

classifier = load_model()

text = st.text_input("Enter text to analyze sentiment:")

if st.button("Analyze"):
    if text.strip():
        results = classifier(text, top_k=None)

    for result in results:
        emotion = result["label"]
        score = result["score"]

        st.write(emotion.capitalize())
        st.progress(score)

    else:
        st.warning("Please enter some text to analyze.")