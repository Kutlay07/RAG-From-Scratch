from tokenization.tokenizer import Tokenizer
from models.chunk import Chunk
from typing import List
from models.document import Document

class Chunker:
    def __init__(self, tokenizer: Tokenizer, chunk_size: int, overlap: int=0,):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap

        if chunk_size <= 0:
            raise ValueError("Chunk Size should be greater than zero.")
        if overlap < 0:
            raise ValueError("Overlap cannot be negative.")
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size.")

    def split(self, document: Document) -> List[Chunk]:
        token_ids = self.tokenizer.encode(document.text)
        chunks = []

        start = 0
        index = 0
        
        while start < len(token_ids):
            end = start + self.chunk_size
            current_token_ids = token_ids[start:end]
            text = self.tokenizer.decode(current_token_ids)
            chunk = Chunk(
                            text=text,
                            token_ids=current_token_ids,
                            document=document,
                            chunk_index=index)
            
            chunks.append(chunk)
            start = end - self.overlap
            index += 1
            
        return chunks
