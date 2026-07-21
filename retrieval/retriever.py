from typing import List
from embeddings.base_embedder import BaseEmbedder
from vector_store.base_vector_store import BaseVectorStore
from models.chunk import Chunk

class Retriever:
    
    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
        
    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        query_embedding = self.embedder.embed_text(query)
        
        chunks = self.vector_store.search(
            query_embedding = query_embedding,
            top_k = top_k
        )
        return chunks