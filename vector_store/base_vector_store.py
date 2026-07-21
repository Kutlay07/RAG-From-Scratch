from abc import ABC, abstractmethod
from typing import List
from models.chunk import Chunk
import numpy as np


class BaseVectorStore(ABC):
    
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Chunk]:
        pass