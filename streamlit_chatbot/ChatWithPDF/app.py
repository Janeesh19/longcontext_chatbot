import streamlit as st
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

# Securely load OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]  # Replace with your OpenAI API key

# Sales coach prompt and Creta context
system_prompt = """
Act as an expert sales coach. Your job is to answer queries from sales agents to help them sell Hyundai Creta only. 
Use the following 7-step framework to guide your responses:

1. Identify the genuine problems or needs the vehicle meets for customers.
2. Acknowledge other competitive vehicles customers may be considering.
3. Describe how the Hyundai Creta ideally improves customers' lives.
4. Present yourself as a trusted advisor aiming to match the right vehicle to each customer.
5. Explain the main benefits and value the Hyundai Creta provides.
6. Provide transparent evidence to back up claims about the vehicle's features.
7. Offer next steps for interested customers to learn more or test drive.

Never recommend any other car other than Hyundai Creta.
If you do not know the answer to a question, simply state so. 
Focus on being helpful, honest, and customer-oriented in crafting sales coaching advice.
"""

# Replace this variable with the actual context for Hyundai Creta
creta_context = """
(Creta E MT Petrol, Overview, Product Label, 1.5 PL 6MT)
(Creta E MT Petrol, Engine, Engine Label, 1.5 PL 6MT)
(Creta E MT Petrol, Engine, Displacement (cc), 1497)
(Creta E MT Petrol, Engine, Max. Power (ps / rpm), 115 / 6300)
(Creta E MT Petrol, Engine, Max. Torque (Nm / rpm), 143.8 / 4500)
(Creta E MT Petrol, Transmission, Transmission Type, 6 Speed MT)
(Creta E MT Petrol, Fuel Consumption, Fuel Type, Petrol)
(Creta E MT Petrol, Wheels & Tires, Front Tires, 205/65 R16)
(Creta E MT Petrol, Wheels & Tires, Rear Tires, 205/65 R16)
(Creta E MT Diesel, Overview, Product Label, 1.5 DSL 6MT)
(Creta E MT Diesel, Engine, Engine Label, 1.5 DSL 6MT)
(Creta E MT Diesel, Engine, Displacement (cc), 1493)
(Creta E MT Diesel, Engine, Max. Power (ps / rpm), 116 / 4000)
(Creta E MT Diesel, Engine, Max. Torque (Nm / rpm), 250 / 1500~2750)
(Creta E MT Diesel, Transmission, Transmission Type, 6 Speed MT)
(Creta E MT Diesel, Fuel Consumption, Fuel Type, Diesel)
(Creta E MT Diesel, Wheels & Tires, Front Tires, 205/65 R16)
(Creta E MT Diesel, Wheels & Tires, Rear Tires, 205/65 R16)
"""

# ChatPromptTemplate configuration
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    SystemMessagePromptTemplate.from_template(creta_context),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Chat model and memory
llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.7)
memory = ConversationBufferMemory(return_messages=True, memory_key="history")
conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=True)

# Main Streamlit application
def main():
    st.set_page_config(page_title="Chat with Sales Coach 🚗", page_icon="🚗")

    # Initialize session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = conversation
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    st.header("Chat with Sales Coach 🚗")

    # Add a Clear Chat button
    clear_chat = st.button("Clear Chat")
    if clear_chat:
        # Clear chat history and reset session state
        st.session_state.chat_history = []
        st.session_state.user_input = ""
        st.rerun()

    # Input box for user's question
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Ask your question:",
            value=st.session_state.user_input,
            key="dynamic_user_input"
        )
    with col2:
        send_button = st.button("Send")

    if send_button:
        if user_input.strip():
            # Run the conversation
            response = st.session_state.conversation.run({"input": user_input})
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Reset user input
            st.session_state.user_input = ""

            # Replace experimental_rerun with st.rerun
            st.rerun()
        else:
            st.warning("Please enter a valid question.")

    # Display chat history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
