from ingestion.directory_loader import DirectoryLoader
from ingestion.loader_factory import LoaderFactory
from ingestion.arxiv_loader import ArxivLoader
from tokenization.tokenizer import Tokenizer
from ingestion.chunker import Chunker
from embeddings.huggingface_embedder import HuggingFaceEmbedder
from vector_store.faiss_vector_store import FAISSVectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.cross_encoder_reranker import CrossEncoderReranker
from llm.huggingface_llm import HuggingFaceLLM
from pipeline.rag_pipeline import RAGPipeline
from pathlib import Path
from cache.embedding_cache import EmbeddingCache

import config


def main():
    
    loader_factory = LoaderFactory()
    directory_loader = DirectoryLoader(loader_factory)
    arxiv_loader = ArxivLoader()
    
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
    
    bm25_retriever = BM25Retriever()
    
    hybrid_retriever = HybridRetriever(
        vector_retriever=retriever,
        bm25_retriever=bm25_retriever,
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
        arxiv_loader=arxiv_loader,
        loader_factory=loader_factory,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        llm=llm,
    )
    
    index_path = Path(config.VECTOR_STORE_PATH)

    if (index_path / "index.faiss").exists():
        print("Loading existing vector database...")
        vector_store.load(config.VECTOR_STORE_PATH)

        hybrid_retriever.build_index(
            vector_store.chunks
        )

    else:
        print("Building vector database...")

        rag.ingest_directory(config.DOCUMENTS_PATH)

        if config.USE_ARXIV:
            rag.ingest_arxiv(
                query=config.ARXIV_QUERY,
                max_results=config.ARXIV_MAX_RESULTS,
            )

        vector_store.save(config.VECTOR_STORE_PATH)

        hybrid_retriever.build_index(
            vector_store.chunks
        )
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
