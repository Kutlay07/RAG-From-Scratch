from typing import List
from pathlib import Path
from tqdm import tqdm

from ingestion.text_loader import TextLoader
from models.document import Document


class DirectoryLoader:

    def __init__(self, text_loader: TextLoader):
        self.text_loader = text_loader

    def load(self, directory: str) -> List[Document]:
        documents = []

        files = list(Path(directory).rglob("*.txt"))

            
        for file in tqdm(files, desc="Loading documents"):
            document = self.text_loader.load(file)
            documents.append(document)
            
        print(f"Successfully loaded {len(documents)} documents.")

        return documents