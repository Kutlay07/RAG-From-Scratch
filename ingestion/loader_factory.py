from pathlib import Path

from ingestion.text_loader import TextLoader
from ingestion.pdf_loader import PDFLoader
from ingestion.markdown_loader import MarkdownLoader
from ingestion.html_loader import HTMLLoader


class LoaderFactory:
    
    def __init__(self):
        self._loaders = {
            ".txt": TextLoader(),
            ".pdf": PDFLoader(),
            ".md": MarkdownLoader(),
            ".html": HTMLLoader(),
            ".htm": HTMLLoader(),
        }
        
    def get_loader(self, path: Path):
        suffix = path.suffix.lower()
        
        loader = self.loaders.get(suffix)
        
        if loader is None:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        return loader