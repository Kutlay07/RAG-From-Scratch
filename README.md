# RAG-From-Scratch

A modular Retrieval-Augmented Generation (RAG) system implemented from scratch using Hugging Face models with clean software architecture principles.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow?logo=huggingface)
![Status](https://img.shields.io/badge/Status-Ongoing-blue)

</p>

> 🚧 This project is actively under development. New features are added incrementally through versioned releases.

---

## Architecture

<p align="center">

![RAG Architecture](images/rag-architecture.png)

</p>

---

## Features

- Modular RAG architecture
- Multi-format document ingestion
- TXT, PDF, Markdown, and HTML loaders
- Recursive directory loading
- Token-based chunking
- Hugging Face sentence embeddings
- Embedding cache
- FAISS vector search
- Persistent FAISS index
- Cross-Encoder reranking
- Prompt template system
- Hugging Face LLM integration
- Source references in answers
- Dependency Injection architecture
- Easily extensible design

---

## Project Structure

```text
RAG-From-Scratch
│
├── cache/
│   └── embedding_cache.py
│
├── data/
│   └── documents/
│
├── embeddings/
├── generation/
│   ├── prompt_builder.py
│   └── prompts/
│       └── default.txt
│
├── images/
├── ingestion/
│   ├── text_loader.py
│   ├── pdf_loader.py
│   ├── markdown_loader.py
│   ├── html_loader.py
│   ├── directory_loader.py
│   └── loader_factory.py
│
├── llm/
├── models/
├── output/
│   ├── cache/
│   └── faiss/
│
├── pipeline/
├── retrieval/
├── tokenization/
├── vector_store/
│
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Kutlay07/RAG-From-Scratch.git

cd RAG-From-Scratch
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Add your documents into:

```text
data/documents/
```

Supported formats:

```text
.txt
.pdf
.md
.html
```

2. Configure the models and parameters:

```python
config.py
```

3. Run:

```bash
python main.py
```

---

## Current Pipeline

```text
Documents
    ↓
Loader Factory
    ↓
Document Loader
    ↓
Chunker
    ↓
Embedding Cache
    ↓
Embedder
    ↓
FAISS Vector Store
    ↓
Retriever
    ↓
Cross Encoder Reranker
    ↓
Prompt Template
    ↓
LLM
    ↓
Response + Sources
```

---

## Roadmap

### v1.0 ✅

- End-to-end RAG pipeline
- Hugging Face embeddings
- Cross Encoder reranking
- Brute-force vector store

### v1.1 ✅

- Recursive directory ingestion
- Progress tracking
- Document statistics
- Source references

### v1.2 ✅

- PDF Loader
- Markdown Loader
- HTML Loader
- Loader Factory
- Multi-format document ingestion

### v1.3 ✅

- FAISS vector store
- Persistent vector database

### v1.4 ✅

- Embedding cache
- Prompt template system
- Automatic FAISS reuse

### v2.0 🚀

- ArXiv Loader
- BM25 Search
- Hybrid Search
- Conversation Memory
- Chat UI
- Agentic RAG

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- tiktoken
- pypdf
- BeautifulSoup4

---

## Demo

Coming soon...

- Chat UI
- Agentic RAG
- Hybrid Search

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.