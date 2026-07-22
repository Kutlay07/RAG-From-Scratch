from ingestion.directory_loader import DirectoryLoader
from ingestion.loader_factory import LoaderFactory
from tokenization.tokenizer import Tokenizer
from ingestion.chunker import Chunker
from embeddings.huggingface_embedder import HuggingFaceEmbedder
from vector_store.faiss_vector_store import FAISSVectorStore
from retrieval.retriever import Retriever
from retrieval.cross_encoder_reranker import CrossEncoderReranker
from llm.huggingface_llm import HuggingFaceLLM
from pipeline.rag_pipeline import RAGPipeline
from pathlib import Path
from cache.embedding_cache import EmbeddingCache

import config


def main():
    
    loader_factory = LoaderFactory()
    directory_loader = DirectoryLoader(loader_factory)
    
    cache = EmbeddingCache(cache_dir=config.CACHE_PATH)
    
    tokenizer = Tokenizer(config.TOKENIZER_ENCODING)

    chunker = Chunker(
        tokenizer,
        config.CHUNK_SIZE,
        config.OVERLAP,
    )

    embedder = HuggingFaceEmbedder(
        model_name=config.EMBEDDING_MODEL,
        device=config.DEVICE,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        cache=cache
    )

    vector_store = FAISSVectorStore(dimension=config.EMBEDDING_DIMENSION)
    
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store
    )
    
    reranker = CrossEncoderReranker(
        model_name=config.RERANKER_MODEL,
        device=config.DEVICE,
        batch_size=config.RERANK_BATCH_SIZE,
    )
    
    llm = HuggingFaceLLM(
        model_name=config.LLM_MODEL,
        device=config.DEVICE,
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
    )
    
    rag = RAGPipeline(
        directory_loader=directory_loader,
        loader_factory=loader_factory,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        retriever=retriever,
        reranker=reranker,
        llm=llm,
    )
    
    index_path = Path(config.VECTOR_STORE_PATH)


    if (index_path / "index.faiss").exists():
        print("Loading existing FAISS index...")
        vector_store.load(config.VECTOR_STORE_PATH)

    else:
        print("Building FAISS index...")
        rag.ingest_directory(config.DOCUMENTS_PATH)
        vector_store.save(config.VECTOR_STORE_PATH)

    answer, chunks = rag.ask(
        query=config.QUERY,
        top_k=config.TOP_K,
    )

    print(answer)
    print("\nSources:")

    seen = set()

    for chunk in chunks:
        key = (chunk.document.title, chunk.document.source)

        if key not in seen:
            seen.add(key)

            print(f"- {chunk.document.title}")
            print(f"  {chunk.document.source}")


if __name__ == "__main__":
    main()
