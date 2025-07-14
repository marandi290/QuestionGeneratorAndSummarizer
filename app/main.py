import streamlit as st
from parser import extract_text_from_pdf
from summarizer import generate_summary
from question_generator import generate_mcqs

st.title("📘 Lecture Notes QnA Summarizer")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.subheader("📄 Extracted Text")
    st.write(text[:1000] + "...")  # Preview

    if st.button("Generate Summary and Questions"):
        with st.spinner("Generating..."):
            summary = generate_summary(text)
            questions = generate_mcqs(text)

        st.subheader("📝 Summary")
        st.write(summary)
        st.subheader("❓ MCQs")
        for q in questions:
            st.write(q)