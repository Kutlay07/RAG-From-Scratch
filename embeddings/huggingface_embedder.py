from embeddings.base_embedder import BaseEmbedder
from sentence_transformers import SentenceTransformer
from typing import List
from models.chunk import Chunk

class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, model_name, device, batch_size):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name).to(device)
        
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        texts = [(chunk.text) for chunk in chunks]
        embeddings = self.model.encode(texts,
                        batch_size=self.batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=True)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        return chunks