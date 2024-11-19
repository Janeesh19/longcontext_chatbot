import streamlit as st
import PyPDF2
import openai
import requests

# Securely access API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
grok_api_key = st.secrets["GROK_API_KEY"]

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chat_with_gpt(prompt, context):
    response = openai.chat_completion(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "You are an assistant knowledgeable about the uploaded PDF."},
            {"role": "assistant", "content": context},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

# ... rest of your code remains the same ...
