from llm.base_llm import BaseLLM
from generation.prompt_builder import load_prompt


class QueryRewriter:
    
    def __init__(self, llm:BaseLLM):
        self.llm = llm
    
    
    def rewrite(self, query: str, history: str) -> str:
        template = load_prompt("query_rewriter")
        
        prompt = template.format(
            history = history,
            query = query,
        )
    
        rewritten_query = self.llm.generate(prompt)
        
        return rewritten_query.strip()