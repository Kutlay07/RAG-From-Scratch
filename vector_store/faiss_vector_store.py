from typing import List
import numpy as np
import faiss
import pickle
from pathlib import Path

from vector_store.base_vector_store import BaseVectorStore
from models.chunk import Chunk


class FAISSVectorStore(BaseVectorStore):

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []


    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        embeddings = np.array(
            [chunk.embedding for chunk in chunks],
            dtype="float32"
        )

        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        self.chunks.extend(chunks)


    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Chunk]:

        query_embedding = query_embedding.astype("float32")

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(
                query_embedding,
                axis=0
            )

        faiss.normalize_L2(query_embedding)

        _, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for idx in indices[0]:
            if idx != -1:
                results.append(self.chunks[idx])

        return results


    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(path / "index.faiss")
        )

        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)


    def load(self, path: str) -> None:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Storage path not found: {path}"
            )

        self.index = faiss.read_index(
            str(path / "index.faiss")
        )

        with open(path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)