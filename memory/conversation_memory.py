





class ConversationMemory:
    
    def __init__(self):
        self.messages = []


    def add_user_message(self, message: str):
        
        self.messages.append({
            "role": "user",
            "content": message,
        })
        
        
    def add_assistant_message(self, message: str):
        
        self.messages.append({
            "role": "assistant",
            "content": message,
        })
        
    def get_context(self) -> str:
        context = []
        
        for message in self.messages:
            role = message["role"].capitalize()
            content = message["content"]
            
            context.append(f"{role}:\n{content}")
            
        return "\n\n".join(context)
        
    def clear(self):
        self.messages.clear()
        
        