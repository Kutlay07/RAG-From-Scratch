# RAG-From-Scratch

A modular Retrieval-Augmented Generation (RAG) system implemented from scratch using Hugging Face models and clean software architecture principles.

---

## Architecture

<p align="center">

![RAG Architecture](images/rag-architecture.png)

</p>

---

## Features

- Modular RAG architecture
- Text document ingestion
- Token-based chunking
- Hugging Face sentence embeddings
- Brute-force vector search
- Cross-Encoder reranking
- Prompt construction
- Hugging Face LLM integration
- Dependency Injection architecture
- Easily extensible design

---

## Project Structure

```text
RAG-From-Scratch
│
├── data/
├── embeddings/
├── generation/
├── images/
├── ingestion/
├── llm/
├── models/
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

```bash
git clone https://github.com/Kutlay07/RAG-From-Scratch.git

cd RAG-From-Scratch

pip install -r requirements.txt
```

---

## Usage

1. Put your documents inside the `data/documents/` directory.

2. Configure the models in `config.py`.

3. Run:

```bash
python main.py
```

---

## Current Pipeline

```text
Documents
    ↓
Loader
    ↓
Chunker
    ↓
Embedder
    ↓
Vector Store
    ↓
Retriever
    ↓
Reranker
    ↓
Prompt Builder
    ↓
LLM
    ↓
Response
```

---

## Roadmap

### v1.0

- ✅ End-to-end RAG pipeline
- ✅ Hugging Face Embeddings
- ✅ Cross Encoder Reranking
- ✅ Brute Force Vector Store

### v1.1

- ⬜ Directory Loader
- ⬜ PDF Loader
- ⬜ Markdown Loader
- ⬜ Better Prompt Templates

### v1.2

- ⬜ FAISS Vector Store
- ⬜ Embedding Cache
- ⬜ Streaming Responses

### v2.0

- ⬜ Hybrid Search
- ⬜ Conversation Memory
- ⬜ Evaluation Pipeline
- ⬜ Agentic RAG

---

## License

MIT License