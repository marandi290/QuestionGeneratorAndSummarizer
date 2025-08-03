import os
from PyPDF2 import PdfReader
from app.modules.text_cleaning import clean_text

def extract_and_clean_uploaded_file(file) -> str:
    text = ''
    filename = file.filename.lower()
    print("filename: ", filename)
    if filename.endswith('.txt'):
        file_content = file.read().decode('utf-8')
        text = clean_text(file_content)

    elif filename.endswith('.pdf'):
        reader = PdfReader(file)
        pdf_text = ''
        for page in reader.pages:
            pdf_text += page.extract_text() or ''
        text = clean_text(pdf_text)

    else:
        raise ValueError("Unsupported file type. Please upload a .txt or .pdf file.")

    return text
