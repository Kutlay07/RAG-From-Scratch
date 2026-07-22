from embeddings.base_embedder import BaseEmbedder
from sentence_transformers import SentenceTransformer
from typing import List
from models.chunk import Chunk
import numpy as np
from cache.embedding_cache import EmbeddingCache

class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, device: str,
                batch_size: int, cache: EmbeddingCache):
        
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name).to(device)
        self.cache = cache
        
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        texts_to_embed = []
        chunks_to_embed = []
        
        for chunk in chunks:
            if self.cache.exists(chunk.text):
                chunk.embedding = self.cache.load(chunk.text)
            else:
                texts_to_embed.append(chunk.text)
                chunks_to_embed.append(chunk)
                
        if texts_to_embed:
            embeddings = self.model.encode(
                texts_to_embed,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            for chunk, embedding in zip(chunks_to_embed, embeddings):
                chunk.embedding = embedding
                self.cache.save(chunk.text, embedding)
        return chunks
    
    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )