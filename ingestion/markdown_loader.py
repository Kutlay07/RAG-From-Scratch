from pathlib import Path

from models.document import Document


class MarkdownLoader:

    def load(self, path: Path) -> Document:
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"File not found:\n{path}")

        if path.suffix.lower() != ".md":
            raise ValueError("Expected a Markdown file.")

        content = path.read_text(encoding="utf-8")

        return Document(
            text=content,
            source=str(path),
            title=path.stem,
        )