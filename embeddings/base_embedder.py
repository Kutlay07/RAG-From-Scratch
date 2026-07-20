from typing import List
from abc import ABC, abstractmethod
from models.chunk import Chunk


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        pass