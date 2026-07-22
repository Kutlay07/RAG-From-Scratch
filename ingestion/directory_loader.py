from typing import List
from pathlib import Path
from tqdm import tqdm

from ingestion.loader_factory import LoaderFactory
from models.document import Document


class DirectoryLoader:

    def __init__(self, loader_factory: LoaderFactory):
        self.loader_factory = loader_factory

    def load(self, directory: str) -> List[Document]:
        documents = []

        files = [f for f in Path(directory).rglob("*") if f.is_file()]

            
        for file in tqdm(files, desc="Loading documents"):
            loader = self.loader_factory.get_loader(file)
            try:
                document = loader.load(file)
                documents.append(document)

            except Exception as e:
                print(f"Skipping {file}: {e}")
                
        print(f"Successfully loaded {len(documents)} documents.")

        return documents