from retrieval.retrieval import load_corpus, embed_texts, find_top_k

class Paper:
    def __init__(self, text, chunk_size=5) -> None:
        self.text = text
        self.chunks = []
        sentences = self.text.split('. ')
        for i in range(0, len(sentences), chunk_size):
            self.chunks.append('. '.join(sentences[i:i+chunk_size]))
        # use e5 model to embed each chunk of the paper
        self.embed_paper()

    def embed_paper(self):
        self.emb = embed_texts(self.chunks)
        return self.emb
    
    def retrieve_top_k(self, query, k=5):
        return find_top_k(query, self.emb, k=k)