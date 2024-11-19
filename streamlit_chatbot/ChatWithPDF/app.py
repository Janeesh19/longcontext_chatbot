import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import requests

# Securely load API keys
openai_api_key = st.secrets["OPENAI_API_KEY"]
grok_api_key = st.secrets["GROK_API_KEY"]

# Function to call Grok API
def call_grok(api_key, prompt, max_tokens=300, temperature=0.7):
    endpoint = "https://api.grok.ai/v1/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "model": "grok-v1",
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    response = requests.post(endpoint, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json().get("completion", "")
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")


# Function to process PDFs and extract text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    if not text.strip():
        st.warning("No readable text was found in the uploaded PDF.")
    return text


# Split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)


# Create FAISS vectorstore from text chunks
def get_vectorstore(text_chunks):
    if not text_chunks:
        st.error("No text chunks available for processing. Please upload a valid PDF.")
        return None
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# Get conversational chain for GPT
def get_conversation_chain(vectorstore, model_name):
    if model_name == "GPT":
        llm = ChatOpenAI(api_key=openai_api_key)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever(), memory=memory
        )
    return None


# Handle user input
def handle_userinput(user_question, selected_model, vectorstore):
    if selected_model == "GPT":
        if not st.session_state.conversation:
            st.error("Please upload a PDF and add data before asking questions.")
            return
        response = st.session_state.conversation({"question": user_question})
        if response and "answer" in response:
            st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
        else:
            st.error("Failed to get a response from GPT.")
    elif selected_model == "Grok":
        try:
            response = call_grok(grok_api_key, user_question)
            if response:
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            else:
                st.error("Grok returned an empty response.")
        except Exception as e:
            st.error(f"Grok API Error: {str(e)}")


# Main Streamlit application
def main():
    st.set_page_config(page_title="Chat with PDF :books:", page_icon=":books:")

    # Initialize session states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "temp_input" not in st.session_state:
        st.session_state.temp_input = ""

    st.header("Chat with PDF :books:")

    # Model selection dropdown
    selected_model = st.selectbox(
        "Choose a model:", options=["GPT", "Grok"], index=0
    )

    # Input box for user's question with Send button
    col1, col2 = st.columns([4, 1])  # Split space for input and button
    with col1:
        user_input = st.text_input(
            "Ask your question:",
            value=st.session_state.temp_input,  # Use session state to manage input value
            key="user_input",  # Unique key for the widget
        )
    with col2:
        send_button = st.button("Send")  # Button to submit the question

    # Process the input when "Send" is clicked
    if send_button:
        if user_input.strip():  # Ensure the input is not empty or whitespace
            # Save the current question in session state
            st.session_state.current_question = user_input.strip()

            # Add the user's question to chat history
            st.session_state.chat_history.append({"role": "user", "content": st.session_state.current_question})

            # Process the question
            handle_userinput(st.session_state.current_question, selected_model, None)

            # Clear the input field dynamically
            st.session_state.temp_input = ""  # Reset the input box
        else:
            st.warning("Please enter a valid question.")

    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            role = "You" if message["role"] == "user" else "Assistant"
            st.write(f"**{role}:** {message['content']}")
    else:
        st.info("No chat history yet. Start by asking a question!")


if __name__ == "__main__":
    main()
