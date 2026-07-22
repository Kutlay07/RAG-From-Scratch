from typing import List
from pathlib import Path

from ingestion.text_loader import TextLoader
from models.document import Document


class DirectoryLoader:

    def __init__(self, text_loader: TextLoader):
        self.text_loader = text_loader

    def load(self, directory: str) -> List[Document]:
        documents = []

        files = Path(directory).rglob("*.txt")

        for file in files:
            document = self.text_loader.load(file)
            documents.append(document)

        return documents