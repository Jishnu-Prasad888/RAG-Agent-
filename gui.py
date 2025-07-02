import streamlit as st
from streamlit_chat import message
from main import api
# Page config
st.set_page_config(page_title="Sthairya", page_icon="💬")

st.title("ChatGPT-like Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to a wonderland for curiosities 😍"},
        {"role": "assistant", "content": "How may I help you !!!"}
    ]

# Display chat messages
for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["role"] == "user", key=str(i))

# Input box
prompt = st.chat_input("Ask away !!! (press q to exit)")

# When user submits a message
if prompt:
    if prompt == "q":
        st.session_state.messages.append({"role": "assistant", "content": "Sorry to see you gooo 🥺🥹😭😭 "})
        exit()
    else :    
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = api(prompt)
        if response :
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "No response received"})
