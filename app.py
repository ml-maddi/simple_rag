import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# --- CONFIGURATION ---
# Make sure to set your GOOGLE_API_KEY as an environment variable
# or Streamlit secret (st.secrets["GOOGLE_API_KEY"])
try:
    # For Streamlit Community Cloud, set this in the secrets management
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, AttributeError):
    # For local development, use an environment variable
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set it as an environment variable or Streamlit secret.")
    st.stop()

# --- CONSTANTS ---
PDF_FILE_PATH = "HSC26_Bangla_1st_paper.pdf" # Place your PDF in the same directory
VECTOR_DB_PATH = "chroma_db_multilingual"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
LLM_MODEL_NAME = "gemini-2.5-flash"

# --- PROMPT TEMPLATE FOR LANGUAGE CONSISTENCY ---
prompt_template = """You are a helpful AI assistant for answering questions about a given document.
You are given a question and a set of document chunks as context.
You must strictly follow these rules:
1. Your final answer MUST be in the same language as the user's question.
2. If the user's question is in English, your answer must be in English.
3. If the user's question is in Bengali, your answer must be in Bengali.
4. Carefully analyze the provided context. If the information to answer the question is not in the context, you MUST respond with one of the following sentences, matching the language of the question:
   - For English questions: "Sorry, I am unable to answer this question."
   - For Bengali questions: "দুঃখিত, আপনার প্রশ্নটির উত্তর আমার জানা নেই।"
5. Do not make up answers. Your response must be grounded in the provided context.
6. Base your answer ONLY on the context provided below.

Context:
{context}

Question: {question}
Answer:"""

# --- HELPER FUNCTIONS ---

@st.cache_resource
def load_and_embed_pdf():
    """
    Loads the PDF, splits it into chunks, creates embeddings,
    and stores them in a Chroma vector database.
    This function is cached to avoid re-running on every interaction.
    """
    if os.path.exists(VECTOR_DB_PATH):
        st.info("Loading existing vector database...")
        return Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        )

    if not os.path.exists(PDF_FILE_PATH):
        st.error(f"PDF file not found at {PDF_FILE_PATH}. Please upload the file.")
        with open(PDF_FILE_PATH, "w") as f:
            f.write("Dummy PDF content. Please replace with the actual file.")
        st.stop()

    st.info(f"Loading and processing '{PDF_FILE_PATH}'... This may take a moment.")
    
    loader = PyMuPDFLoader(PDF_FILE_PATH)
    documents = loader.load()

    for doc in documents:
        doc.page_content = doc.page_content.replace('\n', ' ').strip()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        st.error("Could not split the document into chunks. Check the PDF content.")
        return None

    st.info("Creating vector embeddings... This is a one-time process.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    st.success("Vector database created and persisted successfully!")
    return vector_store

@st.cache_resource
def initialize_llm_and_chain(_vector_store):
    """
    Initializes the LLM and the conversational retrieval chain with a custom prompt.
    """
    if _vector_store is None:
        return None, None

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.1, convert_system_message_to_human=True)

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    retriever = _vector_store.as_retriever(search_kwargs={"k": 70})
    
    # Set up the custom prompt
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT} # Injecting the custom prompt
    )
    return chain, memory


# --- STREAMLIT UI ---

st.set_page_config(page_title="Multilingual RAG Chatbot", page_icon="📚")

# --- FONT FIX FOR BENGALI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;700&display=swap');
    html, body, [class*="st-"], [class*="css-"] {
        font-family: 'Noto Sans Bengali', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📚 Multilingual RAG Chatbot (English & বাংলা)")
st.caption("Powered by Google Gemini, LangChain, and HuggingFace")

with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions about the provided document.
    
    - **Corpus:** HSC Bangla 1st Paper
    - **Languages:** English, Bengali
    - **Embedding:** `Qwen/Qwen3-Embedding-0.6B`
    - **LLM:** `gemini-2.5-flash`
    """)
    
    st.header("Sample Questions")
    st.markdown("""
    - অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
    - কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
    - বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
    - What was Kalyani's age during the wedding?
    """)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        if 'memory' in st.session_state:
            st.session_state.memory.clear()
        st.rerun()

try:
    vector_store = load_and_embed_pdf()
    if vector_store:
        chain, memory = initialize_llm_and_chain(vector_store)
        st.session_state.chain = chain
        st.session_state.memory = memory
    else:
        st.session_state.chain = None
except Exception as e:
    st.error(f"An error occurred during initialization: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question in English or Bengali..."):
    if st.session_state.chain is None:
        st.error("The RAG chain is not initialized. Please check the PDF file and configuration.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chain_input = {"question": prompt, "chat_history": st.session_state.get("chat_history", [])}
                result = st.session_state.chain(chain_input)
                response = result['answer']
                
                st.markdown(response)

                with st.expander("View Retrieved Context"):
                    for doc in result['source_documents']:
                        st.info(f"Source: Page {doc.metadata.get('page', 'N/A')}")
                        st.write(doc.page_content)

                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
                error_message = "Sorry, I encountered an error. Please try again."
                st.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
