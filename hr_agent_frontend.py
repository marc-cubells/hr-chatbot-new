import streamlit as st
from hr_agent_backend import get_response

def process_input(user_input):
    response = get_response(user_input)
    return response

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with your HR Bot ", page_icon=":speech_balloon:")
    st.header("Chat with your HR Bot  💬")
    st.info("Type your query in the chat window")

    # Initialize the Streamlit chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input and display the chatbot's response
    if user_input := st.chat_input("Ask your questions from the HTML files"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        result = process_input(user_input)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown(result + "|")

        message_placeholder.markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})


if __name__ == "__main__":
    main()            