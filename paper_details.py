class Paper:
    def __init__(self, text) -> None:
        self.text = text
        self.chunk()
    
    def chunk(self, chunk_size=5):
        self.chunks = []
        sentences = self.text.split('.')
        for i in range(0, len(sentences), chunk_size):
            self.chunk.append('. '.join(sentences[i:i+chunk_size]))