from typing import List
import arxiv
from models.document import Document


class ArxivLoader:

    def load(self, query: str, max_results: int=5) -> List[Document]:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        
        client = arxiv.Client()
        
        results = client.results(search)
        
        documents = []
        
        for result in results:
            document = Document(
            text=result.summary,
            title=result.title,
            source=result.pdf_url,
            metadata={
                "authors": [author.name for author in result.authors],
                "published": result.published.isoformat(),
                "entry_id": result.entry_id,
                },
            )
    
            documents.append(document)
            
        return documents