import os
import streamlit as st
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from PyPDF2 import PdfReader
import tiktoken  # Token counting
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
import validators
from bs4 import BeautifulSoup

# Load API Key
load_dotenv()

# Initialize Streamlit App
st.title("📄 AI-Powered Summarizer (PDF, YouTube & Web URLs)")

# Initialize session state for model selection
if "models" not in st.session_state:
    st.session_state.models = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "last_api_key" not in st.session_state:
    st.session_state.last_api_key = None

# **User Inputs for API Key & Model Selection**
api_key = st.sidebar.text_input(placeholder="Enter Groq API Key", label="🔑 Enter Groq API Key", type="password")

# Fetch available models only if API key changes or models haven't been fetched
if api_key and (not st.session_state.models or st.session_state.last_api_key != api_key):
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        data = response.json()

        if "data" in data:
            st.session_state.models = [f"{model['id']} ({model['owned_by']})" for model in data["data"] if model.get("active", False)]
            
            # Set default model only if not already set or if API key changed
            if not st.session_state.selected_model and st.session_state.models:
                st.session_state.selected_model = st.session_state.models[0]
            
            # Store current API key
            st.session_state.last_api_key = api_key
        else:
            st.error(f"⚠️ Failed to fetch models: {data.get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Request Failed: {e}")

# Model selection function that updates session state
def update_selected_model():
    st.session_state.selected_model = st.session_state.model_dropdown

# Display model selection dropdown only if models are available
if st.session_state.models:
    # Find index of currently selected model
    current_index = 0
    if st.session_state.selected_model in st.session_state.models:
        current_index = st.session_state.models.index(st.session_state.selected_model)
    
    # Model selection with callback
    st.sidebar.selectbox(
        "🤖 Choose LLM Model:", 
        st.session_state.models, 
        index=current_index,
        key="model_dropdown",
        on_change=update_selected_model
    )
    
    # Extract model ID from selection (remove the owner part)
    model_id = st.session_state.selected_model.split(" ")[0] if st.session_state.selected_model else None
else:
    model_id = None
    st.sidebar.warning("⚠️ No models available. Check your API key.")

# **Customization Options**
summary_styles = {
    "Concise": "Summarize the following text in a brief and clear manner.",
    "Detailed": "Summarize with full details, covering all key points.",
    "Technical": "Summarize while maintaining technical terminology and depth.",
    "Creative": "Summarize using a storytelling approach with engaging language."
}
selected_style = st.selectbox("📜 Choose Summary Style:", list(summary_styles.keys()))

temperature = st.slider("🔥 Adjust Model Creativity (Temperature)", 0.0, 1.5, 0.7, 0.1)

### **Step 1: Summarization Function**
def call_summarize_chain(docs):
    if not model_id:
        st.error("❌ No model selected. Please choose an LLM model.")
        return

    llm = ChatGroq(model=model_id, groq_api_key=api_key, temperature=temperature)

    prompt_template = f"""{summary_styles[selected_style]}
    {{text}}
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to refine the given summary.\n"
        "Existing Summary: {existing_answer}\n"
        "You can improve clarity, conciseness, and readability.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Refine the summary with a better structure and readability."
    )
    refine_prompt = PromptTemplate.from_template(refine_template)

    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
    )

    result = chain.invoke({"input_documents": docs}, return_only_outputs=True)
    summary_text = result['output_text']

    st.subheader("📌 AI-Generated Summary:")
    st.write(summary_text)

    if st.button("♻️ Rephrase Summary"):
        refined_summary = llm.invoke(refine_prompt.format(existing_answer=summary_text, text="\n".join([doc.page_content for doc in docs])))
        st.subheader("🔄 Rephrased Summary:")
        st.write(refined_summary)
        summary_text = refined_summary  

    if summary_text:
        st.download_button(
            label="📥 Download Summary as .txt",
            data=summary_text,
            file_name="summary.txt",
            mime="text/plain",
        )

### **Step 2: PDF Summarization**
uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])

if uploaded_file:
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp_uploaded.pdf")
    docs = loader.load()

    encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text):
        return len(encoder.encode(text))

    chunk_size = 12000
    chunk_overlap = 2000

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(docs)

    total_doc_tokens = sum(count_tokens(doc.page_content) for doc in split_docs)
    st.write(f"📌 **Total Document Tokens:** {total_doc_tokens}")
    st.write(f"🔹 **Chunk Token Limit:** {chunk_size}, **Overlap:** {chunk_overlap}")
    st.write(f"🔍 **Number of Chunks:** {len(split_docs)}")

    call_summarize_chain(split_docs)

### **Step 3: Web & YouTube Summarization**
def fetch_static_web_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join([p.get_text() for p in soup.find_all("p")])
        return Document(page_content=text) if text.strip() else None
    except Exception as e:
        st.error(f"Error fetching website content: {e}")
        return None

def extract_youtube_id(url):
    parsed_url = urlparse(url)
    if parsed_url.netloc in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    elif parsed_url.netloc in ["youtu.be"]:
        return parsed_url.path.lstrip("/")
    return None

def get_youtube_transcript(video_url):
    video_id = extract_youtube_id(video_url)
    if not video_id:
        st.error("Invalid YouTube URL")
        return None

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([t["text"] for t in transcript])
        return Document(page_content=transcript_text)
    except Exception as e:
        st.error(f"Error fetching YouTube transcript: {e}")
        return None

url = st.text_input("🎥 Enter YouTube or Website URL")  

if st.button("Summarize"):
    if not validators.url(url):
        st.error("🚨 Please enter a valid URL")
    else:
        try:
            with st.spinner("Fetching and summarizing..."):
                doc = None

                if "youtube.com" in url or "youtu.be" in url:
                    doc = get_youtube_transcript(url)
                else:
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False)
                    docs = loader.load()
                    doc = docs[0] if docs else None

                    if not doc or not doc.page_content.strip():
                        doc = fetch_static_web_content(url)

                if not doc or not doc.page_content.strip():
                    st.error("⚠️ No text content found to summarize.")
                    st.stop()

                call_summarize_chain([doc])

        except Exception as e:
            st.error(f"An error occurred: {e}")