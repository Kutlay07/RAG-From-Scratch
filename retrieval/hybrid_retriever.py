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
        
        rrf_scores = {}
        
        k = 60
        
        for rank, chunk in enumerate(vector_results, start=1):
            key = (
                chunk.document.source,
                chunk.chunk_index,
            )
            
            rrf_scores[key] = {
                "chunk": chunk,
                "score": 1 / (k + rank)
            }
            

            
        for rank, chunk in enumerate(bm25_results, start=1):
            key = (
                chunk.document.source,
                chunk.chunk_index
                )
            
            score = 1 / (k + rank)
            
            if key in rrf_scores:
                rrf_scores[key]["score"] += score
            else:
                rrf_scores[key] = {
                    "chunk": chunk,
                    "score": score,
                }
                
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda item: item["score"],
            reverse=True,
        )
        
        return [item["chunk"] for item in sorted_results[:top_k]]