class Paper:
    def __init__(self, text) -> None:
        self.text = text
        self.chunk()
    
    def chunk(self):
        self.chunks = None
        raise NotImplemented