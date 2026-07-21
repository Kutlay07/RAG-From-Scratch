from abc import ABC, abstractmethod
from typing import List
from models.chunk import Chunk


class BaseReranker(ABC):
    
    @abstractmethod
    def rerank(self, query: str, chunks: List[Chunk], top_k: int) -> List[Chunk]:
        pass