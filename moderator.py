class Moderator:
    def __init__(self, model_id):
        self.model = None # define model - Llama 3.
    
    def is_expand(self, prev_args, new_args):
        return False