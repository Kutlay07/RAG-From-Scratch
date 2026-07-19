from dataclasses import dataclass, field
from typing import List
from models.document import Document

@dataclass
class Chunk:
    text: str
    token_ids: List[int]
    document: Document
    chunk_index: int