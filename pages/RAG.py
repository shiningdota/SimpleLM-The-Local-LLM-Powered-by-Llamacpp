import streamlit as st
from openai import OpenAI
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import nltk
from nltk.tokenize import sent_tokenize

# Setting the document processing functions
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_txt(file):
    return file.read().decode("utf-8")

def read_docx(file):
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Setting chunks for the document
def create_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 20) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            if chunk_overlap > 0 and current_chunk:
                overlap_size = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    if overlap_size + len(sent) <= chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += len(sent)
                    else:
                        break
                current_chunk = overlap_sentences
                current_size = sum(len(sent) for sent in current_chunk)
            else:
                current_chunk = []
                current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Setting for generate embedding and set prefixed_text
def generate_embeddings(chunks: List[str], embedding_model) -> np.ndarray:
    embeddings = []
    for chunk in chunks:
        prefixed_text = f"search_document: {chunk}"
        embedding = embedding_model.encode(prefixed_text)
        embeddings.append(embedding)
    return np.array(embeddings)

def find_most_relevant_chunks(question: str, chunks: List[str], chunk_embeddings: np.ndarray, embedding_model, top_k: int = 3) -> List[str]:
    question_embedding = embedding_model.encode(f"search_query: {question}")
    similarities = np.dot(chunk_embeddings, question_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# Set combine for document upload (pdf, text, docx)
def process_document(uploaded_file, embedding_model):
    if uploaded_file.name.endswith('.pdf'):
        text = read_pdf(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        text = read_txt(uploaded_file)
    elif uploaded_file.name.endswith('.docx'):
        text = read_docx(uploaded_file)
    else:
        st.error("Unsupported file type.")
        return None, None, None
    
    chunks = create_chunks(text, chunk_size=1000, chunk_overlap=20)
    chunk_embeddings = generate_embeddings(chunks, embedding_model)
    return text, chunks, chunk_embeddings

# Setting the LLM API
def load_llm(base_url, api_key):
    return OpenAI(base_url=base_url, api_key=api_key)

# Setting the embedding model and caching
@st.cache_resource
def load_embedding_model():
    device = "cpu"
    return SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True).to(device)

# Streamlit UI - Set Page
st.set_page_config(page_title="Local RAG Chatbot", layout="wide")
st.title("Local RAG Chatbot")

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
        llm_prompt = st.text_input("Prompt", value="You are an assistant designed to answer questions based on provided documents. Use the following information to respond accurately to the user's question.", type="default")
        llm_temperature = st.slider("Temperature", 0.0, 1.0, TEMPERATURE)
        llm_frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, FREQ_PENALTY)
        llm_max_tokens = st.slider("Max Tokens", 128, 4096, MAX_TOKENS, 128)
        llm_top_p = st.slider("Top P", 0.0, 2.0, TOP_P)
    
    st.header("Document Settings")
    uploaded_file = st.file_uploader("Upload Document (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])
    top_k = st.slider("Number of relevant chunks", min_value=1, max_value=5, value=3)
    
    if st.button("Reset Chat"):
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! Do you have any questions regarding the documents provided?"}]
        st.session_state.pop("processed_chunks", None)
        st.session_state.pop("chunk_embeddings", None)
        st.session_state.pop("full_text", None)

# Streamlit UI - Idle State
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! Do you have any questions regarding the documents provided?"}]

# Streamlit UI - Load embedding and document uploader
embedding_model = load_embedding_model()

if uploaded_file is not None:
    if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        with st.spinner("Processing document..."):
            full_text, chunks, chunk_embeddings = process_document(uploaded_file, embedding_model)
            if chunks is not None:
                st.session_state.processed_chunks = chunks
                st.session_state.chunk_embeddings = chunk_embeddings
                st.session_state.full_text = full_text
                st.success("Document processed successfully!")

# Streamlit UI - History Chat
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# Streamlit UI - LLM settings parameters
client = load_llm(llm_base_url, llm_api_key)

def get_llm_response(question, context=None):
    messages = [{"role": "system", "content": llm_prompt}]
    
    if context:
        messages.append({"role": "system", "content": f"Use this context to answer the question: {context}"})
    
    messages.append({"role": "user", "content": question})
    
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
chat_input = st.chat_input("Ask a question about the document")

if chat_input:
    st.session_state.messages.append({"role": "user", "content": chat_input})
    st.chat_message("user").write(chat_input)
    
    # If document is loaded, use RAG
    if hasattr(st.session_state, 'processed_chunks') and st.session_state.processed_chunks:
        relevant_chunks = find_most_relevant_chunks(
            chat_input,
            st.session_state.processed_chunks,
            st.session_state.chunk_embeddings,
            embedding_model,
            top_k=top_k
        )
        context = "\n".join(relevant_chunks)
        response = get_llm_response(chat_input, context)
        
        # Optionally show used context
        with st.expander("View relevant context used"):
            st.write(context)
    else:
        # Regular chat without RAG
        response = get_llm_response(chat_input)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)