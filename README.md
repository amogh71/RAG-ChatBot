# RAG-ChatBot
# 📚  RAG-Powered ChatBot 

This project is an **intelligent chatbot** using **Retrieval Augmented Generation (RAG)** to provide instant, accurate answers from your business documents. It's like having a smart assistant for all your company's information.

## ✨ What It Does

* **Understands Your Documents:** Ingests and comprehends PDFs from various business domains (CRM, HR, Finance, etc.), including text and tables.
* **Finds Answers Fast:** Quickly locates the most relevant information within your documents based on your questions.
* **Generates Smart Responses:** Uses **Google Gemini AI** to create detailed, human-like answers grounded in the retrieved context, complete with examples.
* **Connects Diverse Data:** Acts as a central knowledge hub, pulling insights across different operational areas.

## 🚀 How It Works

This system combines several advanced AI techniques:

1.  **Document Processing:** PDFs are processed by `pdfplumber` and broken into chunks.
2.  **Vector Embeddings:** `SentenceTransformers` converts these chunks into numerical "fingerprints" (embeddings).
3.  **Intelligent Retrieval:** These embeddings are stored in **FAISS**, a fast vector database. When you ask a question, FAISS finds the most similar "fingerprints" to provide relevant context.
4.  **AI-Powered Generation:** This context is then fed to the **Gemini 2.0 Flash-Exp LLM**, which generates precise, hallucination-free answers.
5.  **User Interface:** A **Streamlit** app provides an intuitive chat interface for interaction and data management.

## ⚙️ Key Technologies

* **Streamlit:** Web app framework.
* **Google Generative AI (Gemini):** Core LLM.
* **FAISS:** Vector database for similarity search.
* **SentenceTransformers:** For text embeddings.
* **pdfplumber:** PDF content extraction.

## 🏃‍♀️ Get Started

1.  **Clone the repository:** `git clone [YOUR_REPO_URL_HERE]`
2.  **Install dependencies:** `pip install -r requirements.txt` (create this file first: `pip freeze > requirements.txt`)
3.  **Configure API Key:** Replace `YOUR_GEMINI_API_KEY_HERE` in the code with your actual Gemini API key.
4.  **Update PDF Paths:** Adjust the `PDF_DIRECTORY_*` variables to your local PDF locations.
5.  **Run the app:** `streamlit run app.py`


   <img width="2552" height="1352" alt="image" src="https://github.com/user-attachments/assets/512c9548-28fe-4259-b99e-4ec7c6c41301" />
