class Moderator:
    def __init__(self, model):
        self.model = model # define model - Llama 3.
    
    def is_expand(self, debate_node):
        return False