from typing import List

from models.chunk import Chunk
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:
    
    def __init__(self, vector_retriever: Retriever,
                bm25_retriever: BM25Retriever):
        
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        
    def build_index(self, chunks: List[Chunk]):
        self.bm25_retriever.build_index(chunks)
        
        
    def search(self, query: str, top_k: int = 5) -> List[Chunk]:

        candidate_k = top_k * 2
        
        vector_results = self.vector_retriever.retrieve(
            query,
            candidate_k
        )
        
        bm25_results = self.bm25_retriever.search(
            query,
            candidate_k
        )
        
        combined = []
        seen = set()
        
        for chunk in vector_results + bm25_results:

            key = (
                chunk.document.source,
                chunk.chunk_index
            )
            
            if key not in seen:
                seen.add(key)
                combined.append(chunk)
                
        return combined[:top_k]