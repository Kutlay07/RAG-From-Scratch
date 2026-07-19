from pathlib import Path
from models.document import Document

class TextLoader:
    
    def load(self, path: Path) -> Document:
        path = Path(path)
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            raise FileNotFoundError(f"File not found:\n {path}")
        return Document(
            text = content,
            source = str(path),
            title = path.stem
            # path.name -> python.txt
            # path.stem -> python  which is much more beautiful
            )
