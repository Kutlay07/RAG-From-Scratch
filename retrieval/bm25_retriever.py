from typing import List
from rank_bm25 import BM25Okapi
from models.chunk import Chunk
import numpy as np


class BM25Retriever:
    
    def __init__(self):
        self.bm25 = None
        self.chunks = []
        
    
    def build_index(self, chunks: List[Chunk]):
        self.chunks = chunks
        
        tokenized_chunks = [
            chunk.text.lower().split()
            for chunk in chunks
        ]
        
        self.bm25 = BM25Okapi(tokenized_chunks)
        
    
    def search(self, query: str, top_k:int=5) -> List[Chunk]:
        self.query = query
        
        query_tokens = query.lower().split()
        
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            self.chunks[i]
            for i in top_indices
        ]
        
        return results