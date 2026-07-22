from pathlib import Path
from typing import List

from models.chunk import Chunk

PROMPT_DIR = Path(__file__).parent / "prompts"
def load_prompt(name: str) -> str:
    prompt_path = PROMPT_DIR / f"{name}.txt"
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
    
    
def build_prompt(query: str, chunks: List[Chunk]) -> str:

    context = "\n\n".join(
        chunk.text for chunk in chunks
    )
    
    template = load_prompt("default")
    
    prompt = template.format(
        context = context,
        query = query,
    )

    return prompt