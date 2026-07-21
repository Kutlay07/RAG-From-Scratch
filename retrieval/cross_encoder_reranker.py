from typing import List
from models.chunk import Chunk
from retrieval.base_reranker import BaseReranker
from sentence_transformers import CrossEncoder


class CrossEncoderReranker(BaseReranker):
    
    def __init__(self, model_name: str, device: str, batch_size: int):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device)
        
    def rerank(self, query: str, chunks: List[Chunk], top_k: int) -> List[Chunk]:
        pairs = [(query, chunk.text) for chunk in chunks]
        
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        reranked_results = list(zip(scores, chunks))
        
        reranked_results.sort(key=lambda x: x[0], reverse=True)
        
        return [chunk for _, chunk in reranked_results[:top_k]]