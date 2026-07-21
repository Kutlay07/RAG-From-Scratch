from typing import List
from models.chunk import Chunk


def build_prompt(query: str, chunks: List[Chunk]) -> str:
    context = "\n\n".join(
        f"Title: {chunk.document.title}\n"
        f"Source: {chunk.document.source}\n"
        f"Text:\n{chunk.text}"
        for chunk in chunks
    )
    
    prompt = f"""You are a helpful AI assistant.
    
        Answer the question using ONLY the provided context.
        
        If the answer is not in the context, say "I don't know."
        
        Context:
        {context}
        
        Question:
        {query}
        Answer:
        """
    
    return prompt

