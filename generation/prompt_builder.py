from typing import List
from models.chunk import Chunk


def build_prompt(query, chunks):

    context = "\n\n".join(
        chunk.text for chunk in chunks
    )

    prompt = f"""You are a helpful AI assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt