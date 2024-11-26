# Input box for user's question
col1, col2 = st.columns([9, 1])  # Adjusted column proportions for alignment
with col1:
    user_input = st.text_input(
        "Ask your question:",
        value=st.session_state.user_input,
        key="dynamic_user_input",
        placeholder="Type your question and press Enter.",
        on_change=lambda: execute_user_input()  # Execute on Enter
    )

# Function to handle the user input execution
def execute_user_input():
    if st.session_state.dynamic_user_input.strip():
        user_input = st.session_state.dynamic_user_input.strip()
        if not st.session_state.pdf_chunks:
            st.error("Please upload and process a file before asking questions.")
        else:
            try:
                response = st.session_state.conversation.run({"input": user_input})
                st.session_state.chat_history.insert(0, {"role": "user", "content": user_input})
                st.session_state.chat_history.insert(0, {"role": "assistant", "content": response})
                st.session_state.user_input = ""
                st.experimental_rerun()
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Display recent question and answer below the input box
if len(st.session_state.chat_history) >= 2:
    recent_question = st.session_state.chat_history[1]  # Most recent question
    recent_answer = st.session_state.chat_history[0]  # Most recent answer

    st.subheader("Recent Q&A")
    st.markdown(f"**You (👤):** {recent_question['content']}")
    st.markdown(f"**Assistant (🤖):** {recent_answer['content']}")

# Display chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in reversed(st.session_state.chat_history):  # Older messages move down
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
