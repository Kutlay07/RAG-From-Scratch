from vector_store.base_vector_store import BaseVectorStore
import numpy as np
from typing import List
from models.chunk import Chunk

class BruteForceVectorStore(BaseVectorStore):
    
    def __init__(self):
        self.chunks: List[Chunk] = []
        
    def add_chunks(self, chunks):
        self.chunks.extend(chunks)
        
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Chunk]:
        scores = []
        
        for chunk in self.chunks:
            similarity = self._cosine_similarity(chunk.embedding, query_embedding)
            scores.append((similarity, chunk))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        
        return [chunk for _, chunk in scores[:top_k]]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
                    
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)