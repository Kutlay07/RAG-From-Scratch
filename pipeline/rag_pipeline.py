from ingestion.directory_loader import DirectoryLoader
from ingestion.loader_factory import LoaderFactory
from ingestion.chunker import Chunker
from embeddings.base_embedder import BaseEmbedder
from vector_store.base_vector_store import BaseVectorStore
from retrieval.base_reranker import BaseReranker
from llm.base_llm import BaseLLM
from generation.prompt_builder import build_prompt
from models.document import Document
from models.chunk import Chunk
from typing import List, Tuple
from pathlib import Path
from ingestion.arxiv_loader import ArxivLoader
from retrieval.hybrid_retriever import HybridRetriever
from memory.conversation_memory import ConversationMemory

class RAGPipeline:
    
    def __init__(self,
                directory_loader: DirectoryLoader,
                arxiv_loader: ArxivLoader,
                loader_factory: LoaderFactory,
                chunker: Chunker,
                embedder: BaseEmbedder,
                vector_store: BaseVectorStore,
                hybrid_retriever: HybridRetriever,
                reranker: BaseReranker,
                llm: BaseLLM,
                memory: ConversationMemory):
        
        self.directory_loader = directory_loader
        self.arxiv_loader = arxiv_loader
        self.loader_factory = loader_factory
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.hybrid_retriever = hybrid_retriever
        self.all_chunks = []
        self.reranker = reranker
        self.llm = llm
        self.memory = memory
        
        
    def _ingest_document(self, document: Document) -> int:
        chunks = self.chunker.split(document)
        
        chunks = self.embedder.embed_chunks(chunks)
        
        self.vector_store.add_chunks(chunks)
        
        self.all_chunks.extend(chunks)
        
        self.hybrid_retriever.build_index(self.all_chunks)
        
        return len(chunks)
    
    
    def ingest_arxiv(self, query: str, max_results: int=5):
        documents = self.arxiv_loader.load(
            query=query,
            max_results=max_results,
        )
        
        total_chunks = 0
        
        for document in documents:
            total_chunks += self._ingest_document(document)
            
        print(
            f"Loaded {len(documents)} papers. Indexed {total_chunks} ArXiv chunks."
            )
    
    def ingest(self, path: str) -> None:
        path = Path(path)
        loader = self.loader_factory.get_loader(path)
        document = loader.load(path)
        
        self._ingest_document(document)
    
    def ingest_directory(self, directory: str) -> None:
        documents = self.directory_loader.load(directory)
        total_chunks = 0

        for document in documents:
            total_chunks += self._ingest_document(document)

        print(f"Indexed {total_chunks} chunks.")
        
        
    def ask(self, query: str, top_k: int = 5) -> Tuple[str, List[Chunk]]:
        self.memory.add_user_message(query)
        
        chunks = self.hybrid_retriever.search(query, top_k)
        
        chunks = self.reranker.rerank(query, chunks, top_k)
        
        history = self.memory.get_context()
        
        prompt = build_prompt(
            query = query,
            chunks = chunks,
            history = history,
        )
        
        answer = self.llm.generate(prompt)
        
        self.memory.add_assistant_message(answer)
        
        return answer, chunks