# Library
import streamlit as st

# Streamlit UI - Set title
st.set_page_config(page_title="Version / Other", layout="wide")
st.title("Version /  Other")
st.write("Version 1.0 (llamacpp-b4539)")

# Streamlit UI - Set Changelog
st.subheader("Changelog")
changelog = '''
- First release!!!
- Llamacpp version B4539
- Support Mistral model, Llama model, the new powerfull model Deepseek R1 Distill models, and many more
- Implemented Local LLM powered by Llamacpp + OpenAI API
- Implemented Local RAG Document Chatbot with combination of Sentence Transformer
- Customizeable LLM Settings like Prompt for LLM, Temperature, Max tokens (currently only support up to 4096), etc. 
'''

with st.expander("V1.0"):
    st.markdown(changelog)

