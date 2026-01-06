
import streamlit as st
import pickle
import numpy as np


st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)


@st.cache_resource
def load_model():
    vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
    model = pickle.load(open("model/model.pkl", "rb"))
    return vectorizer, model

vectorizer, model = load_model()


st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.stButton>button {
    background-color: #0d6efd;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
.stTextArea textarea {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


st.sidebar.title("ℹ️ About")
st.sidebar.write(
    "This app classifies emails as **Spam** or **Not Spam** "
    "using a Machine Learning model trained on email data."
)


st.markdown("<h1 style='text-align: center;'>📧 Email Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>Paste an email below and click Predict</p>",
    unsafe_allow_html=True
)

email_text = st.text_area(
    "✉️ Email Content",
    height=180,
    placeholder="Enter email text here..."
)


if st.button("🔍 Predict"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        transformed_text = vectorizer.transform([email_text])
        prediction = model.predict(transformed_text)[0]

        
        prob = model.predict_proba(transformed_text)[0]
        confidence = np.max(prob) * 100

        st.divider()

        if prediction == 1:
            st.error(f"🚨 **Spam Email**  \nConfidence: **{confidence:.2f}%**")
        else:
            st.success(f"✅ **Not Spam**  \nConfidence: **{confidence:.2f}%**")
