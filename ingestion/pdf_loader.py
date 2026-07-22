from pathlib import Path
from pypdf import PdfReader
from models.document import Document


class PDFLoader:
    
    def load(self, path: Path) -> Document:
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"File not found:\n {path}")
        
        if path.suffix.lower() != ".pdf":
            raise ValueError("Expected a PDF file.") 
        
        reader = PdfReader(path)
        
        content = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )
        
        return Document(
            text = content,
            source = str(path),
            title = path.stem
            )
