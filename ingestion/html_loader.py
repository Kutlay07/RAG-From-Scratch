from pathlib import Path
from bs4 import BeautifulSoup
from models.document import Document


class HTMLLoader:

    def load(self, path: Path) -> Document:
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"File not found:\n{path}")

        if path.suffix.lower() not in {".html", ".htm"}:
            raise ValueError("Expected an HTML file.")

        html = path.read_text(encoding="utf-8")

        soup = BeautifulSoup(html, "html.parser")

        content = soup.get_text(separator="\n", strip=True)

        return Document(
            text=content,
            source=str(path),
            title=path.stem,
        )