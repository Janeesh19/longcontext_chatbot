st.markdown("""
    <style>
    .user-message {
        background-color: #B3E5FC; /* Light Blue for user messages */
        padding: 10px;
        border-radius: 10px;
        text-align: right;
        margin-left: auto;
        margin-right: 10px;
        max-width: 70%;
        color: #000; /* Black text color */
    }
    .assistant-message {
        background-color: #FFECB3; /* Light Yellow for assistant messages */
        padding: 10px;
        border-radius: 10px;
        text-align: left;
        margin-left: 10px;
        margin-right: auto;
        max-width: 70%;
        color: #000; /* Black text color */
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    .chat-container .chat-message {
        margin: 5px 0;
        display: flex;
    }
    .chat-container .user-message {
        justify-content: flex-end;
    }
    .chat-container .assistant-message {
        justify-content: flex-start;
    }
    </style>
""", unsafe_allow_html=True)
