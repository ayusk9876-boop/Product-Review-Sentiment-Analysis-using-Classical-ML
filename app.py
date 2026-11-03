from pathlib import Path
import joblib, streamlit as st
MODEL=Path(__file__).resolve().parent/"models"/"sentiment_model.joblib"
st.set_page_config(page_title="Product Review Sentiment Analyzer",page_icon="⭐")
st.title("Product Review Sentiment Analyzer")
st.write("Enter a product review to predict its sentiment.")
if not MODEL.exists():
    st.warning("Train the model first: run `python main.py`.")
else:
    model=joblib.load(MODEL)
    review=st.text_area("Product review")
    if st.button("Analyze sentiment"):
        if not review.strip(): st.error("Please enter a review.")
        elif int(model.predict([review])[0])==1: st.success("POSITIVE 😊")
        else: st.error("NEGATIVE ☹️")
