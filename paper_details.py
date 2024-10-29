class Paper:
    def __init__(self, text, chunk_size=2) -> None:
        self.text = text
        self.chunks = []
        sentences = self.text.split('.')
        for i in range(0, len(sentences), chunk_size):
            self.chunks.append('. '.join(sentences[i:i+chunk_size]))