# AI Assistant with RAG System

A fully local AI Assistant powered by Retrieval Augmented Generation (RAG) that lets you chat with your PDF documents using local LLMs.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)

---

## What It Does

- Load any PDF documents into a local vector database
- Ask questions in natural language and get accurate answers
- Cites the exact source document and page number for every answer
- Maintains conversation history for follow-up questions
- Runs 100% locally — your documents never leave your machine

---

## Architecture

```
PDFs → PyPDF Loader → Text Chunker → Embeddings (HuggingFace) → ChromaDB
                                                                      ↓
User Question → Retriever (MMR) → Relevant Chunks → LLM (Ollama) → Answer
```

| Component | Technology |
|---|---|
| Document Loading | LangChain PyPDFDirectoryLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-mpnet-base-v2 |
| Vector Database | ChromaDB |
| LLM | Ollama (llama3.1) |
| UI | Streamlit |
| Retrieval Strategy | MMR (Maximum Marginal Relevance) |

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- NVIDIA GPU recommended (CPU works but slower)

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-assistant-rag.git
cd ai-assistant-rag
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull the LLM model
```bash
ollama pull llama3.1
```

### 5. Add your PDFs
```
ai-assistant-rag/
└── data/
    ├── your-document.pdf
    ├── another-document.pdf
    └── ...
```

### 6. Vectorize your documents
```bash
python vectorize_documents.py
```

> Only needed once. Delete `vector_db_dir/` and re-run whenever you add new PDFs.

### 7. Launch the app
```bash
streamlit run main.py
```

Open your browser at `http://localhost:8501`

---

## Project Structure

```
ai-assistant-rag/
├── data/                    # Place your PDF files here
├── vector_db_dir/           # Auto-generated vector database (git ignored)
├── main.py                  # Streamlit app
├── vectorize_documents.py   # PDF loading, chunking & embedding pipeline
├── requirements.txt         # Python dependencies
└── README.md
```

## 💻 GPU Support

The app automatically detects and uses your GPU for embeddings. A GPU indicator is shown in the sidebar.

For Ollama to use your GPU, make sure you have:
- NVIDIA drivers installed
- CUDA 12.1+ toolkit
- PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Dependencies

```
streamlit
langchain
langchain-community
langchain-text-splitters
langchain-chroma
langchain-huggingface
pypdf
chromadb
sentence-transformers
ollama
torch
```

---

## Sample Data Used for Testing

The system was tested using the **Data Classification Policy** published by the **Saudi Data and Artificial Intelligence Authority (SDAIA)**.

| Field | Details |
|---|---|
| **Document** | Data Classification Policy and Regulations |
| **Publisher** | Saudi Data & AI Authority (SDAIA) |
| **Description** | Sets the framework for classifying data received, produced, or dealt with by public entities, regardless of source, form, or nature |
| **Source** | [sdaia.gov.sa](https://sdaia.gov.sa/en/SDAIA/about/Documents/DataClassificationPolicy.pdf) |
| **Access** | Publicly available |

> This document was used strictly for testing and demonstration purposes. The system is not limited to this document.
