from typing import List
from abc import ABC, abstractmethod
from models.chunk import Chunk
import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        pass