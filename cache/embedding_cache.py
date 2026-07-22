import hashlib
import pickle
from pathlib import Path
import numpy as np

class EmbeddingCache:
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        
    def _get_cache_path(self, text: str) -> Path:
        hash_text = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()
    
        return self.cache_dir / f"{hash_text}.pkl"
    
    def exists(self, text: str) -> bool:
        cache_path = self._get_cache_path(text)
        
        return cache_path.exists()


    def load(self, text: str) -> np.ndarray:
        cache_path = self._get_cache_path(text)
        
        with open(cache_path, "rb") as f:
            embedding = pickle.load(f)
            
        return embedding
    
    def save(self, text: str, embedding: np.ndarray) -> None:
        cache_path = self._get_cache_path(text)
        
        with open(cache_path, "wb") as f:
            pickle.dump(embedding, f)