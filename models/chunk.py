from dataclasses import dataclass
from typing import List
from models.document import Document
import numpy as np

@dataclass
class Chunk:
    text: str
    token_ids: List[int]
    document: Document
    chunk_index: int
    embedding: np.ndarray | None = None