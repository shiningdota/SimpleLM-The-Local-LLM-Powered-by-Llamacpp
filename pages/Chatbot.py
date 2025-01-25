# Library
import streamlit as st
from openai import OpenAI

# Streamlit UI - Set title
st.set_page_config(page_title="LLM Chatbot", layout="wide")
st.title("LLM Chatbot")

# Streamlit UI - LLM Settings in the sidebar
TEMPERATURE = 0.7
FREQ_PENALTY = 1.1
MAX_TOKENS = 2048
TOP_P = 0.9
MODEL = "Local"
BASE_URL = "http://127.0.0.1:8080/v1"
APIKEY = "None"

with st.sidebar:
    st.header("LLM Settings")
    llm_base_url = st.text_input("Input your base url", BASE_URL, key="base_url", type="default")
    llm_api_key = st.text_input("Input your API key", APIKEY, key="api_key", type="default")
    llm_model = st.text_input("Input your Model name", MODEL, key="model", type="default")
    with st.expander("Advanced Settings"):
        llm_prompt = st.text_input("Prompt", value="You are a helpful assistant.", type="default")
        llm_temperature = st.slider("Temperature", 0.0, 1.0, TEMPERATURE)
        llm_frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, FREQ_PENALTY)
        llm_max_tokens = st.slider("Max Tokens", 128, 4096, MAX_TOKENS, 128)
        llm_top_p = st.slider("Top P", 0.0, 2.0, TOP_P)
    if st.button("Reset Chat"):
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you?"}]

# Streamlit UI - Idle State
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you?"}]

# Streamlit UI - History Chat
for response in st.session_state.messages:
    st.chat_message(response["role"]).write(response["content"])

# Streamlit UI - LLM settings parameters
client = OpenAI(base_url=llm_base_url, api_key=llm_api_key)

def llm_settings(question):
    # Set the history chat into context
    messages = [{"role": "system", "content": llm_prompt}] + st.session_state.messages + [{"role": "user", "content": question}]

    # Set OpenAI Api chatbot
    response = client.chat.completions.create(
        model=llm_model,
        messages=messages,
        max_tokens=llm_max_tokens,
        temperature=llm_temperature,
        top_p=llm_top_p,
        frequency_penalty=llm_frequency_penalty
    )
    return response.choices[0].message.content

# Streamlit UI - Chat section
llm_chat = st.chat_input("")

if llm_chat:
    st.session_state.messages.append({"role": "user", "content": llm_chat})
    st.chat_message("user").write(llm_chat)
    response = llm_settings(llm_chat)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
