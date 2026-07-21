from llm.base_llm import BaseLLM
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

class HuggingFaceLLM(BaseLLM):
    
    def __init__(self,model_name: str,device: str,max_new_tokens: int,temperature:float):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt")
        
        inputs = inputs.to(self.device)
        
        is_sampling = self.temperature > 0.0
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=is_sampling)
        
        """
        =if u want to cut the prompt part use generated_tokens instead of outputs[0]=
        
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        """

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )
        
        return response