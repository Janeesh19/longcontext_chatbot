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
    response = openai.chat_completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "You are an assistant knowledgeable about the uploaded PDF."},
            {"role": "assistant", "content": context},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

def chat_with_grok(prompt, context):
    url = 'https://api.grok.ai/v1/chat/completions'  # Replace with actual endpoint
    headers = {
        'Authorization': f'Bearer {grok_api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'grok-large',  # Replace with actual model name
        'messages': [
            {"role": "system", "content": "You are an assistant knowledgeable about the uploaded PDF."},
            {"role": "assistant", "content": context},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return None

# Main App
st.title("PDF Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    st.session_state['pdf_text'] = text

    model_options = ['GPT', 'Grok']
    selected_model = st.selectbox("Select a Model", model_options)

    if 'selected_model' not in st.session_state:
        st.session_state['selected_model'] = None

    if selected_model:
        st.session_state['selected_model'] = selected_model

    if st.session_state['selected_model'] == 'GPT':
        st.header("Chat with GPT")
        user_input = st.text_input("You:", key='gpt_input')
        if user_input:
            response = chat_with_gpt(user_input, st.session_state['pdf_text'])
            if response:
                st.text_area("GPT:", value=response, height=200)

    elif st.session_state['selected_model'] == 'Grok':
        st.header("Chat with Grok")
        user_input = st.text_input("You:", key='grok_input')
        if user_input:
            response = chat_with_grok(user_input, st.session_state['pdf_text'])
            if response:
                st.text_area("Grok:", value=response, height=200)
