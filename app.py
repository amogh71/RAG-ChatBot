import streamlit as st
import os
import sys
import pickle
import google.generativeai as genai

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# # # # ------------------- API Configuration -------------------
api_key = "AIzaSyAdkWBs1PPN1YRHurI8B2VNgJHM-WiYJuM"
genai.configure(api_key=api_key)

# generation_config = {
#     "temperature": 1,
#     "top_p": 0.95,
#     "top_k": 40,
#     "max_output_tokens": 8192,
#     "response_mime_type": "text/plain",
# }

# # Initialize LLM
# model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp", generation_config=generation_config)
# chat_session = model.start_chat(history=[])

# # ------------------- FAISS and Embedding Model Setup -------------------
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# dimension = 384  
# index = faiss.IndexFlatL2(dimension)
# doc_store = {}

# # ------------------- Paths for Persistent Storage -------------------
# FAISS_INDEX_PATH = "faiss_index.index"
# DOC_STORE_PATH = "doc_store.pkl"

# # ------------------- Memory Usage Calculation -------------------
# def get_memory_usage():
#     """Calculate memory usage of FAISS index and document store."""
#     faiss_size = sys.getsizeof(pickle.dumps(index))  # Serialize FAISS index to get size
#     doc_store_size = sys.getsizeof(pickle.dumps(doc_store))  # Serialize doc_store to get size
#     return faiss_size, doc_store_size

# # ------------------- PDF Processing -------------------
# def extract_text_from_pdf(pdf_path):
#     """Extract text from a PDF file."""
#     try:
#         with open(pdf_path, 'rb') as pdf_file:
#             pdf_reader = PyPDF2.PdfReader(pdf_file)
#             text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
#             return text.strip()
#     except Exception as e:
#         print(f"Error reading the PDF: {e}")
#         return None

# def store_text_in_faiss(text, doc_id):
#     """Chunk text, create embeddings, and store in FAISS."""
#     global index, doc_store

#     chunk_size = 500
#     overlap_size = 100
#     chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap_size)]

#     embeddings = embedding_model.encode(chunks)
#     index.add(np.array(embeddings, dtype=np.float32))

#     for i, chunk in enumerate(chunks):
#         doc_store[len(doc_store)] = {"text": chunk, "doc_id": doc_id}

# def summarize_and_store(pdf_path, doc_id):
#     """Extract, summarize, and store PDF text."""
#     pdf_text = extract_text_from_pdf(pdf_path)
#     if pdf_text:
#         store_text_in_faiss(pdf_text, doc_id)
#         return pdf_text
#     return None

# # ------------------- Retrieval and LLM Interaction -------------------
# def retrieve_relevant_text(query, k=5):
#     """Fetch relevant text chunks from FAISS."""
#     global index, doc_store
#     query_embedding = embedding_model.encode([query]).astype(np.float32)
#     distances, indices = index.search(query_embedding, k)

#     relevant_texts = [doc_store[idx]["text"] for idx in indices[0] if idx in doc_store]
#     return " ".join(relevant_texts) if relevant_texts else "No relevant information found."

# def ask_llm_with_context(user_query):
#     """Retrieve relevant info and ask LLM."""
#     context = retrieve_relevant_text(user_query)
#     prompt = f"Using the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {user_query}"
    
#     response = chat_session.send_message(prompt)
#     return response.text

# # ------------------- Persist FAISS Index and Document Store -------------------
# def save_faiss_index():
#     """Save FAISS index to disk."""
#     faiss.write_index(index, FAISS_INDEX_PATH)

# def load_faiss_index():
#     """Load FAISS index from disk if it exists."""
#     global index
#     if os.path.exists(FAISS_INDEX_PATH):
#         index = faiss.read_index(FAISS_INDEX_PATH)

# def save_doc_store():
#     """Save document store to disk."""
#     with open(DOC_STORE_PATH, "wb") as f:
#         pickle.dump(doc_store, f)

# def load_doc_store():
#     """Load document store from disk if it exists."""
#     global doc_store
#     if os.path.exists(DOC_STORE_PATH):
#         with open(DOC_STORE_PATH, "rb") as f:
#             doc_store = pickle.load(f)

# # ------------------- Streamlit Web App -------------------
# def clear_data():
#     """Clear FAISS index and document store."""
#     global index, doc_store
#     index = faiss.IndexFlatL2(dimension)
#     doc_store = {}
#     save_faiss_index()
#     save_doc_store()

# def main():
#     st.title("PDF Text Extraction and LLM Query")

#     load_faiss_index()
#     load_doc_store()

#     option = st.sidebar.selectbox("Choose an option", ["Upload PDF", "Ask Question", "Clear Data"])

#     if option == "Upload PDF":
#         uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
#         if uploaded_file is not None:
#             pdf_path = os.path.join("uploaded_files", uploaded_file.name)
#             os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
#             with open(pdf_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())

#             doc_id = f"document_{len(doc_store) + 1}"
#             summarize_and_store(pdf_path, doc_id)
#             save_faiss_index()
#             save_doc_store()
#             st.success("PDF processed and stored successfully.")

#     elif option == "Ask Question":
#         user_query = st.text_input("Enter your question:")
#         if user_query:
#             answer = ask_llm_with_context(user_query)
#             st.write("Answer:", answer)

#     elif option == "Clear Data":
#         if st.button("Clear All Data"):
#             clear_data()
#             st.success("All data has been cleared.")

#     # Display FAISS and document store memory usage
#     faiss_size, doc_store_size = get_memory_usage()
#     st.sidebar.write(f"FAISS Index Size: {faiss_size / (1024 * 1024):.2f} MB")
#     st.sidebar.write(f"Document Store Size: {doc_store_size / (1024 * 1024):.2f} MB")

#     st.sidebar.text("Developed with Streamlit")

# if __name__ == "__main__":
#     main()






# import streamlit as st
# import os
# import sys
# import pickle
# import google.generativeai as genai
# import PyPDF2
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer

# # ------------------- API Configuration -------------------
# api_key = "AIzaSyAdkWBs1PPN1YRHurI8B2VNgJHM-WiYJuM"
# genai.configure(api_key=api_key)

# generation_config = {
#     "temperature": 1,
#     "top_p": 0.95,
#     "top_k": 40,
#     "max_output_tokens": 8192,
#     "response_mime_type": "text/plain",
# }

# # Initialize LLM
# model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp", generation_config=generation_config)
# chat_session = model.start_chat(history=[])

# # ------------------- FAISS and Embedding Model Setup -------------------
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# dimension = 384  
# index = faiss.IndexFlatL2(dimension)
# doc_store = {}

# # ------------------- Paths for Persistent Storage -------------------
# FAISS_INDEX_PATH = "faiss_index.index"
# DOC_STORE_PATH = "doc_store.pkl"

# # ------------------- Memory Usage Calculation -------------------
# def get_memory_usage():
#     """Calculate memory usage of FAISS index and document store."""
#     faiss_size = sys.getsizeof(pickle.dumps(index))
#     doc_store_size = sys.getsizeof(pickle.dumps(doc_store))
#     return faiss_size, doc_store_size

# # ------------------- PDF Processing -------------------
# def extract_text_from_pdf(pdf_path):
#     """Extract text from a PDF file."""
#     try:
#         with open(pdf_path, 'rb') as pdf_file:
#             pdf_reader = PyPDF2.PdfReader(pdf_file)
#             text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
#             return text.strip()
#     except Exception as e:
#         st.error(f"Error reading the PDF: {e}")
#         return None

# def store_text_in_faiss(text, doc_id):
#     """Chunk text, create embeddings, and store in FAISS."""
#     global index, doc_store

#     chunk_size = 500
#     overlap_size = 100
#     chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap_size)]

#     embeddings = embedding_model.encode(chunks)
#     index.add(np.array(embeddings, dtype=np.float32))

#     for i, chunk in enumerate(chunks):
#         doc_store[len(doc_store)] = {"text": chunk, "doc_id": doc_id}

# def summarize_and_store(pdf_path, doc_id):
#     """Extract, summarize, and store PDF text."""
#     pdf_text = extract_text_from_pdf(pdf_path)
#     if pdf_text:
#         store_text_in_faiss(pdf_text, doc_id)
#         return pdf_text
#     return None

# # ------------------- Retrieval and LLM Interaction -------------------
# def retrieve_relevant_text(query, k=5):
#     """Fetch relevant text chunks from FAISS."""
#     global index, doc_store
#     query_embedding = embedding_model.encode([query]).astype(np.float32)
#     distances, indices = index.search(query_embedding, k)

#     relevant_texts = [doc_store[idx]["text"] for idx in indices[0] if idx in doc_store]
#     return " ".join(relevant_texts) if relevant_texts else "No relevant information found."

# def ask_llm_with_context(user_query):
#     """Retrieve relevant info and ask LLM."""
#     context = retrieve_relevant_text(user_query)
#     prompt = f"Using the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {user_query}"
    
#     response = chat_session.send_message(prompt)
#     return response.text

# # ------------------- Persist FAISS Index and Document Store -------------------
# def save_faiss_index():
#     """Save FAISS index to disk."""
#     faiss.write_index(index, FAISS_INDEX_PATH)

# def load_faiss_index():
#     """Load FAISS index from disk if it exists."""
#     global index
#     if os.path.exists(FAISS_INDEX_PATH):
#         index = faiss.read_index(FAISS_INDEX_PATH)

# def save_doc_store():
#     """Save document store to disk."""
#     with open(DOC_STORE_PATH, "wb") as f:
#         pickle.dump(doc_store, f)

# def load_doc_store():
#     """Load document store from disk if it exists."""
#     global doc_store
#     if os.path.exists(DOC_STORE_PATH):
#         with open(DOC_STORE_PATH, "rb") as f:
#             doc_store = pickle.load(f)

# # ------------------- Clear Data -------------------
# def clear_data():
#     """Clear FAISS index and document store."""
#     global index, doc_store
#     index = faiss.IndexFlatL2(dimension)
#     doc_store = {}
#     save_faiss_index()
#     save_doc_store()

# # ------------------- Streamlit Web App -------------------
# def main():
#     st.title("PDF Text Extraction and LLM Query")

#     # Initialize session state
#     if 'current_page' not in st.session_state:
#         st.session_state.current_page = 'upload'

#     # Load existing data
#     load_faiss_index()
#     load_doc_store()

#     # Sidebar buttons
#     st.sidebar.header("Actions")
#     col1, col2, col3 = st.sidebar.columns(3)
#     with col1:
#         if st.button("📤 Upload PDF"):
#             st.session_state.current_page = 'upload'
#     with col2:
#         if st.button("❓ Ask Question"):
#             st.session_state.current_page = 'ask'
#     with col3:
#         if st.button("🧹 Clear Data"):
#             st.session_state.current_page = 'clear'

#     # Main content area
#     if st.session_state.current_page == 'upload':
#         st.header("Upload PDF Document")
#         uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
#         if uploaded_file is not None:
#             pdf_path = os.path.join("uploaded_files", uploaded_file.name)
#             os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
#             with open(pdf_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())

#             doc_id = f"document_{len(doc_store) + 1}"
#             summarize_and_store(pdf_path, doc_id)
#             save_faiss_index()
#             save_doc_store()
#             st.success("PDF processed and stored successfully!")

#     elif st.session_state.current_page == 'ask':
#         st.header("Ask Questions")
#         user_query = st.text_input("Enter your question:")
#         if user_query:
#             answer = ask_llm_with_context(user_query)
#             st.subheader("Answer:")
#             st.write(answer)

#     elif st.session_state.current_page == 'clear':
#         st.header("Clear Data")
#         st.warning("This will permanently delete all stored data!")
#         if st.button("Confirm Permanent Deletion"):
#             clear_data()
#             st.success("🧹 All data has been successfully cleared!")

#     # Display memory usage
#     faiss_size, doc_store_size = get_memory_usage()
#     st.sidebar.markdown("---")
#     st.sidebar.subheader("Storage Usage")
#     st.sidebar.write(f"FAISS Index: {faiss_size / (1024 * 1024):.2f} MB")
#     st.sidebar.write(f"Document Store: {doc_store_size / (1024 * 1024):.2f} MB")
#     st.sidebar.markdown("---")
    
# if __name__ == "__main__":
#     main()








import streamlit as st
import os
import sys
import pickle
import google.generativeai as genai

import faiss
import numpy as np
import pdfplumber
import pandas as pd
from sentence_transformers import SentenceTransformer


# ------------------- API Configuration -------------------
api_key = "AIzaSyAdkWBs1PPN1YRHurI8B2VNgJHM-WiYJuM"
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize LLM
model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp", generation_config=generation_config)
chat_session = model.start_chat(history=[])

# ------------------- FAISS and Embedding Model Setup -------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384  
index = faiss.IndexFlatL2(dimension)
doc_store = {}

# ------------------- Paths for Persistent Storage -------------------
FAISS_INDEX_PATH = "faiss_index.index"
DOC_STORE_PATH = "doc_store.pkl"
#PDF_DIRECTORY_DB ="/Users/kiran/Documents/ChatBotRagLocal/DATABASE"
# PDF_DIRECTORY_CRM =r"/Users/kiran/Desktop/ChatbotRag/DATABASE/FinancialData"
# PDF_DIRECTORY_HR =r"/Users/kiran/Desktop/ChatbotRag/DATABASE/HR"
# PDF_DIRECTORY_SUPPY_CHAIN =r"/Users/kiran/Desktop/ChatbotRag/DATABASE/SupplyChain"
# PDF_DIRECTORY_SHEET =r"/Users/kiran/Desktop/ChatbotRag/DATABASE/Financialdata"
# PDF_DIRECTORY_MINERAL_PROCESSING=r"/Users/kiran/Desktop/ChatbotRag/DATABASE/Mineral"

PDF_DIRECTORY_CRM =r"C:\Users\amogh\Desktop\ChatbotRag\PDF\dd\CRM"
PDF_DIRECTORY_HR =r"C:\Users\amogh\Desktop\ChatbotRag\PDF\dd\HR"
PDF_DIRECTORY_SUPPY_CHAIN =r"C:\Users\amogh\Desktop\ChatbotRag\PDF\dd\SUPPLY CHAIN"
PDF_DIRECTORY_SHEET =r"C:\Users\amogh\Desktop\ChatbotRag\PDF\dd\SHEET"
PDF_DIRECTORY_MINERAL_PROCESSING=r"C:\Users\amogh\Desktop\ChatbotRag\PDF\dd\MINERAL PROCESSING"
# ------------------- Memory Usage Calculation -------------------
def get_memory_usage():
    """Calculate memory usage of FAISS index and document store."""
    faiss_size = sys.getsizeof(pickle.dumps(index))
    doc_store_size = sys.getsizeof(pickle.dumps(doc_store))
    return faiss_size, doc_store_size

def extract_paragraphs_and_tables(pdf_path):
    """Extract both paragraphs and tables from PDF using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        paragraphs = []
        tables = []
        for page in pdf.pages:
            # Extract text
            text = page.extract_text()
            if text:
                paragraphs.append(text)
            
            # Extract tables
            page_tables = page.extract_tables()
            for table in page_tables:
                if table and len(table) > 1:  # Skip empty tables
                    tables.append(table)
        return paragraphs, tables
    
def load_pdfs_from_directory(PDF_DIRECTORY):
    """Load and process all PDFs from the specified directory."""
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIRECTORY, pdf_file)
        doc_id = f"document_{len(doc_store) + 1}"
        summarize_and_store(pdf_path, doc_id)


def process_pdf_content(pdf_path):
    """Process PDF into cleaned text chunks and table chunks."""
    paragraphs, tables = extract_paragraphs_and_tables(pdf_path)
    
    # Clean text content by removing table data
    cleaned_paragraphs = []
    table_data = set()
    
    # Collect all table cell values
  
    # Remove table data from paragraphs
    for para in paragraphs:
        if para:
            cleaned_para = para
            for data in table_data:
                cleaned_para = cleaned_para.replace(data, "")
            cleaned_paragraphs.append(cleaned_para.strip())
    
    return cleaned_paragraphs

# ------------------- Enhanced Chunking Logic -------------------
def create_chunks(paragraphs):
    """Create chunks with different strategies for text and tables."""
    chunks = []
    
    # Text chunks with overlap
    chunk_size = 500
    overlap = 100
    for text in paragraphs:
        if text:
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start += (chunk_size - overlap)
    
    # Table chunks (row-wise)
    
    
    return chunks

# ------------------- Modified Store Function -------------------
def store_text_in_faiss(pdf_path, doc_id):
    """Process and store PDF content with enhanced chunking."""
    global index, doc_store
    
    # Extract and process content
    paragraphs= process_pdf_content(pdf_path)
    
    # Create chunks
    chunks = create_chunks(paragraphs)
    # Generate and store embeddings
    if chunks:
        embeddings = embedding_model.encode(chunks)
        index.add(np.array(embeddings, dtype=np.float32))
        
        # Store metadata
        for idx, chunk in enumerate(chunks):
            doc_store[len(doc_store)] = {
                "text": chunk,
                "doc_id": doc_id,
                "type": "text" if idx < len(paragraphs) else "table"
            }

# ------------------- Modified Summarize Function -------------------
def summarize_and_store(pdf_path, doc_id):
    """Handle PDF processing and storage."""
    try:
        store_text_in_faiss(pdf_path, doc_id)
        return True
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return False


# ------------------- Retrieval and LLM Interaction -------------------
# def retrieve_relevant_text(query, k=5):
#     """Fetch relevant text chunks from FAISS."""
#     global index, doc_store
#     query_embedding = embedding_model.encode([query]).astype(np.float32)
#     distances, indices = index.search(query_embedding, k)

#     relevant_texts = [doc_store[idx]["text"] for idx in indices[0] if idx in doc_store]
#     return " ".join(relevant_texts) if relevant_texts else "flag"
# def ask_llm_with_context(user_query):
#     """Retrieve relevant info and ask LLM."""
#     context = retrieve_relevant_text(user_query)
    

#     if context == "flag" :
#         prompt = f"answer in detail and give example of the same while ellobrating \n\nQuestion: {user_query}"
    
#         response = chat_session.send_message(prompt)
#     else:

#       prompt = f"Using the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {user_query}"
    
#       response = chat_session.send_message(prompt)
#     return response.text
# ------------------- Retrieval and LLM Interaction -------------------
def retrieve_relevant_text(query, k=5):
    """Fetch relevant text chunks from FAISS."""
    global index, doc_store
    query_embedding = embedding_model.encode([query]).astype(np.float32)
    distances, indices = index.search(query_embedding, k)

    relevant_texts = [doc_store[idx]["text"] for idx in indices[0] if idx in doc_store]
    return " ".join(relevant_texts) if relevant_texts else "flag"

def ask_llm_with_context(user_query):
    """Retrieve relevant info and ask LLM."""
    context = retrieve_relevant_text(user_query)
    
    if context == "flag":
        prompt = (
            """The following is a detailed and well-explained answer to the user's query. 
            Ensure that the response is comprehensive, includes relevant examples, and provides 
            a deep understanding of the topic, even if direct data is not available. Use logical reasoning "
            and prior knowledge to answer.\n\nQuestion: {user_query}\n\nAnswer:"""
        )
    else:
        prompt = (
            f"Using the following context, answer the question in a detailed and comprehensive manner. "
            f"If necessary, extrapolate beyond the context to provide a well-rounded response with examples. "
            f"The response should be clear, informative, and structured."
            f"\n\nContext:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"
        )
    
    response = chat_session.send_message(prompt)
    return response.text









def hr_fn():
    # st.write("Hit: I'm in HR function")
    pass

def balance_sheet_fn():
    pass
    # st.write("Hit: I'm in Balance Sheet function")

def supply_chain_fn():
    pass
    # st.write("Hit: I'm in Supply Chain function")

def prediction_fn():
    pass
    # st.write("Hit: I'm in Prediction function")

              

                    
# ------------------- Persist FAISS Index and Document Store -------------------
def save_faiss_index():
    """Save FAISS index to disk."""
    faiss.write_index(index, FAISS_INDEX_PATH)

def load_faiss_index():
    """Load FAISS index from disk if it exists."""
    global index
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)

def save_doc_store():
    """Save document store to disk."""
    with open(DOC_STORE_PATH, "wb") as f:
        pickle.dump(doc_store, f)

def load_doc_store():
    """Load document store from disk if it exists."""
    global doc_store
    if os.path.exists(DOC_STORE_PATH):
        with open(DOC_STORE_PATH, "rb") as f:
            doc_store = pickle.load(f)

# ------------------- Clear Data -------------------
def clear_data():
    """Clear FAISS index and document store."""
    global index, doc_store
    index = faiss.IndexFlatL2(dimension)
    doc_store = {}
    save_faiss_index()
    save_doc_store()


# Add this at the top with other imports
import streamlit.components.v1 as components

# ------------------- Custom CSS -------------------
def inject_custom_css():
    st.markdown("""
    <style>
        /* Main container */
        .main {
            background-color:rgb(248, 250, 254);
        }
        
        /* Headers */
        h1, h2, h3 {
            color:rgb(220, 232, 235) !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(145deg, #2d3436 0%, #000000 100%) !important;
            color: white !important;
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(145deg, #0984e3, #6c5ce7);
            color: white !important;
            border: none;
            border-radius: 10px;
            padding: 10px 24px;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            border: 2px dashed #6c5ce7;
            border-radius: 15px;
            padding: 20px;
        }
        
        /* Progress bars */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #6c5ce7 0%, #0984e3 100%);
        }
        
        /* Chat bubbles */
        .user-message {
            background:rgb(125, 153, 246);
            border-radius: 15px 15px 0 15px;
            padding: 12px;
            margin: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .bot-message {
            background: #6c5ce7;
            color: white;
            border-radius: 15px 15px 15px 0;
            padding: 12px;
            margin: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# ------------------- Modified Main Function -------------------
def main():
    st.set_page_config(
        layout="wide",
        page_title="C",
        page_icon="📘"
    )
    inject_custom_css()

    # Custom header
    with st.container():
        col1, col2 = st.columns([1, 5])
        with col1:
           st.image("Gemini_Generated_Image_rcx76crcx76crcx7.png", width=80)
        with col2:
            st.title("Chatbot")
            st.caption("Generative Core of Database-ERP-CRM Intelligence")

    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'upload'

    # Load existing data
    load_faiss_index()
    load_doc_store()

    # Enhanced sidebar
    with st.sidebar:
        st.header("Navigation")
        nav_options = {
            "Connect to your DataBase" :'db',
            "Connect to your CRM " :'crm',
            "❓ Ask Questions": "ask",
            "📊 System Health": "stats",
            "🧹 Clear Data": "clear"
        }
        
        for option, page in nav_options.items():
            if st.button(option, use_container_width=True):
                st.session_state.current_page = page
        st.markdown("---")
        
        # System status
        st.subheader("System Status")
        faiss_size, doc_store_size = get_memory_usage()
        
        # Memory gauges
        components.html(f"""
            <div style="background: #2d3436; padding: 15px; border-radius: 10px;">
                <h4 style="color: white; margin-bottom: 15px;">Memory Usage</h4>
                <div style="background: #636e72; height: 8px; border-radius: 4px; margin-bottom: 10px;">
                    <div style="background: #0984e3; width: {(faiss_size/(1024*1024))/2}%; height: 8px; border-radius: 4px;"></div>
                </div>
                <p style="color: white; margin: 0;">FAISS: {faiss_size/(1024*1024):.2f} MB</p>
                
                <div style="background: #636e72; height: 8px; border-radius: 4px; margin: 10px 0;">
                    <div style="background: #6c5ce7; width: {(doc_store_size/(1024*1024))/2}%; height: 8px; border-radius: 4px;"></div>
                </div>
                <p style="color: white; margin: 0;">Documents: {doc_store_size/(1024*1024):.2f} MB</p>
            </div>
        """, height=200)


    if st.session_state.current_page == 'db':       
        # load_pdfs_from_directory(PDF_DIRECTORY_DB)
        save_faiss_index()
        save_doc_store()
        st.markdown("### Choose an option:")
        # Create 4 columns to hold the buttons in one row
        cols = st.columns(5)
        if cols[4].button("   HR   "):
            load_pdfs_from_directory(PDF_DIRECTORY_HR)
            st.success("Database connected successfully!")
            save_faiss_index()
            save_doc_store()

        if cols[0].button("   Mineral Processing  "):
            load_pdfs_from_directory(PDF_DIRECTORY_MINERAL_PROCESSING)
            st.success("Connected to Live plant Data ")
            save_faiss_index()
            save_doc_store()

            # hr_fn()
        if cols[3].button("Finance Data"):
            load_pdfs_from_directory(PDF_DIRECTORY_SHEET)
            st.success("Database connected successfully!")
            save_faiss_index()
            save_doc_store()

            # balance_sheet_fn()
        if cols[1].button("Supply Chain"):
            load_pdfs_from_directory(PDF_DIRECTORY_SUPPY_CHAIN)
            st.success("Database connected successfully!")
            save_faiss_index()
            save_doc_store()
            # supply_chain_fn()
        if cols[2].button("Prediction"):
            # prediction_fn()
            pass
        
    elif st.session_state.current_page == 'crm':       
        load_pdfs_from_directory(PDF_DIRECTORY_CRM)
        save_faiss_index()
        save_doc_store()
        st.success("CRM connected successfully!")


    elif st.session_state.current_page == 'ask':
        st.header("💬 Monitor Plant Process Activities")
        st.markdown("Choose an option ")

        # Custom CSS for button size & spacing
        st.markdown(
            """
            <style>
            /* Ensure uniform button size */
            div[data-testid="stHorizontalBlock"] button {
                width: 160px !important;  /* Same width */
                height: 45px !important;  /* Same height */
                background-color: #4CAF50 !important;  /* Green color */
                color: white !important;
                border-radius: 8px !important;
                border: none !important;
                font-weight: bold !important;
                margin: 8px !important;  /* Adds equal spacing around buttons */
            }

            /* Hover effect */
            div[data-testid="stHorizontalBlock"] button:hover {
                background-color: #45a049 !important; /* Darker green on hover */
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # First row: 3 buttons
        row1 = st.columns([1, 1, 1])  # Three equal-width columns

        with row1[0]:
            if st.button("Finance Data"):
                balance_sheet_fn()

        with row1[1]:
            if st.button("Supply Chain"):
                supply_chain_fn()

        with row1[2]:
            if st.button("Prediction"):
                prediction_fn()

        # Second row: 3 buttons
        row2 = st.columns([1, 1, 1])  # Three equal-width columns

        with row2[0]:
            if st.button("H R"):
                hr_fn()

        with row2[1]:
            if st.button("Mineral"):
                pass

        with row2[2]:
            if st.button("Combined"):
                prediction_fn()

        
            
            
    

    # Initialize session state variables
    if "history" not in st.session_state:
        st.session_state.history = []

    if "image_path" not in st.session_state:
        st.session_state.image_path = None  # Store chart image path

    user_query = st.chat_input("Ask anything about your plants...")

    # Add Clear Chat button
    if st.button("Clear Chat"):
        st.session_state.history = []
        st.session_state.image_path = None  # Clear stored image
        st.rerun()

    chart_keywords = ["chart", "graph", "plot", "visualize", "visualization", "bar chart", "line chart", "histogram", "diagram"]

    # Process user input
    if user_query:
        st.session_state.history.append({"role": "user", "content": user_query})

        if any(keyword in user_query.lower() for keyword in chart_keywords):
            st.session_state.image_path = "/Users/kiran/Desktop/ChatbotRag/output.png"  # Store image path
            st.session_state.history.append({"role": "assistant", "content": "Here is the chart you requested."})
            st.rerun()  # Ensure the image appears immediately

        else:
            with st.spinner("Analyzing documents..."):
                answer = ask_llm_with_context(user_query)
                st.session_state.history.append({"role": "assistant", "content": answer})
            st.session_state.image_path = None  # **Reset image path after text response**
            st.rerun()  # Update chat after generating response

    # Display previous chat messages
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-message'>🙋 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

    # Display the stored image **only if it was recently requested**
    if st.session_state.image_path:
        st.image(st.session_state.image_path, caption="Generated Chart", use_container_width=True)
        st.session_state.image_path = None  # **Reset image path after displaying it**

  


    elif st.session_state.current_page == 'stats':
        st.header("📊 System Analytics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Storage Metrics")
            faiss_size, doc_store_size = get_memory_usage()
            st.metric("Vector Index Size", f"{faiss_size/(1024*1024):.2f} MB")
            st.metric("Document Storage", f"{doc_store_size/(1024*1024):.2f} MB")
            
        with col2:
            st.subheader("Processing Stats")
            st.metric("Total Documents", len(doc_store))
            st.metric("Embedded Chunks", index.ntotal)
            
        st.subheader("Recent Activity")
        activity_data = pd.DataFrame({
            "Time": ["10:00", "10:05", "10:10"],
            "Event": ["Document Uploaded", "Query Processed", "System Maintenance"],
            "Status": ["✅ Success", "⚠️ Warning", "🔧 In Progress"]
        })
        st.dataframe(activity_data, use_container_width=True)

    elif st.session_state.current_page == 'clear':
        st.header("⚠️ Data Management")
        with st.container():
            st.warning("This action will permanently delete all stored data!")

            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                if st.button("🚨 Confirm Full System Reset", use_container_width=True):
                    with st.spinner("Securely erasing data..."):
                        clear_data()
                        st.session_state.history = []
                        st.session_state.current_page = 'upload'  # Reset to upload page
                        st.success("All data successfully erased!")
                        st.rerun()  # Refresh UI immediately


if __name__ == "__main__":
    main()



