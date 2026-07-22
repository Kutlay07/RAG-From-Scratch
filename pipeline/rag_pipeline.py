from ingestion.text_loader import TextLoader
from ingestion.directory_loader import DirectoryLoader
from ingestion.chunker import Chunker
from embeddings.base_embedder import BaseEmbedder
from vector_store.base_vector_store import BaseVectorStore
from retrieval.retriever import Retriever
from retrieval.base_reranker import BaseReranker
from llm.base_llm import BaseLLM
from generation.prompt_builder import build_prompt
from models.document import Document


class RAGPipeline:
    
    def __init__(self,
                loader: TextLoader,
                directory_loader: DirectoryLoader,
                chunker: Chunker,
                embedder: BaseEmbedder,
                vector_store: BaseVectorStore,
                retriever: Retriever,
                reranker: BaseReranker,
                llm: BaseLLM):
        
        self.loader = loader
        self.directory_loader = directory_loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        
    def _ingest_document(self, document: Document) -> None:
        chunks = self.chunker.split(document)
    
        chunks = self.embedder.embed_chunks(chunks)
        
        self.vector_store.add_chunks(chunks)
    
    def ingest(self, path: str) -> None:
        document = self.loader.load(path)
        
        self._ingest_document(document)
    
    def ingest_directory(self, directory: str) -> None:
        documents = self.directory_loader.load(directory)
        for document in documents:
            self._ingest_document(document)

    def ask(self, query: str, top_k: int = 5) -> str:
        # Question -> Retriever -> Reranker -> PromptBuilder -> LLM -> Answer
        chunks = self.retriever.retrieve(query, top_k)
        
        chunks = self.reranker.rerank(query, chunks, top_k)
        
        prompt = build_prompt(query, chunks)
        
        answer = self.llm.generate(prompt)
        
        return answer
    
