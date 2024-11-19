import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Securely load OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

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
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )

# Handle user input
def handle_userinput(user_question, vectorstore):
    if not st.session_state.conversation:
        st.error("Please upload a PDF and add data before asking questions.")
        return
    response = st.session_state.conversation({"question": user_question})
    if response and "answer" in response:
        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
    else:
        st.error("Failed to get a response from GPT.")

# Main Streamlit application
def main():
    st.set_page_config(page_title="Chat with PDF :books:", page_icon=":books:")

    # Initialize session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # Chat Header
    st.header("Chat with PDF :books:")

    # Input Box at the Bottom
    user_input = st.text_input(
        "Ask your question:",
        value=st.session_state.user_input,
        label_visibility="collapsed",
        key="user_input_field",
    )
    send_button = st.button("➤", help="Send", use_container_width=False, key="send_button")

    # Process User Input
    if send_button:
        if user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            handle_userinput(user_input.strip(), st.session_state.conversation)
            st.session_state.user_input = ""  # Clear the input field
            st.experimental_rerun()  # Refresh UI to update the chat
        else:
            st.warning("Please enter a valid question.")

    # Chat History
    if st.session_state.chat_history:
        st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f'<div style="text-align: right; background-color: #FFFFFF; padding: 10px; border-radius: 10px; margin: 5px; max-width: 70%; display: inline-block;">{message["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="text-align: left; background-color: #D6EAF8; padding: 10px; border-radius: 10px; margin: 5px; max-width: 70%; display: inline-block;">{message["content"]}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No chat history yet. Start by asking a question!")

    # Sidebar for Uploading Documents
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Add Data'",
            accept_multiple_files=True
        )
        if st.button("Add Data"):
            with st.spinner("Processing PDFs..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No readable content found in the uploaded PDFs.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        if vectorstore:
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("PDFs have been processed successfully!")
                except Exception as e:
                    st.error(f"Failed to process PDFs: {e}")


if __name__ == "__main__":
    main()
