from ingestion.text_loader import TextLoader
from tokenization.tokenizer import Tokenizer
from ingestion.chunker import Chunker
from embeddings.huggingface_embedder import HuggingFaceEmbedder
from vector_store.brute_force_vector_store import BruteForceVectorStore
from retrieval.retriever import Retriever
from retrieval.cross_encoder_reranker import CrossEncoderReranker
from llm.huggingface_llm import HuggingFaceLLM
from pipeline.rag_pipeline import RAGPipeline
from ingestion.directory_loader import DirectoryLoader
import config


def main():
    
    loader = TextLoader()
    directory_loader = DirectoryLoader(loader)
    
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
    )

    vector_store = BruteForceVectorStore()
    
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
        loader=loader,
        directory_loader=directory_loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        retriever=retriever,
        reranker=reranker,
        llm=llm,
    )
    rag.ingest_directory(config.DOCUMENTS_PATH)

    answer = rag.ask(
        query=config.QUERY,
        top_k=config.TOP_K,
    )

    print(answer)

if __name__ == "__main__":
    main()
    
