import tiktoken
from typing import List

class Tokenizer:
    def __init__(self, encoding):
        self.encoding = tiktoken.get_encoding(encoding)
        
    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        return self.encoding.decode(token_ids)