import json
import numpy as np
from retrieval.e5_model import e5_embed
from typing import List, Dict, Tuple

def load_corpus(path: str) -> List[str]:
    """Load corpus chunks from a JSON file."""
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return list(data.values())
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")

def embed_texts(texts: List[str]) -> Dict[str, np.ndarray]:
    """Embed a list of texts using e5_embed."""
    return e5_embed(texts)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def find_top_k(query: str, corpus_emb: Dict[str, np.ndarray], k: int = 10) -> List[Tuple[str, float]]:
    """Find top-k similar chunks in a corpus to the query."""
    query_emb = e5_embed([query])[query]
    similarities = []
    for chunk, emb in corpus_emb.items():
        sim = cosine_similarity(query_emb, emb)
        similarities.append((chunk, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

def main():
    corpus_paths = {
        # "C99 Segmentation": "chunking/chunks/c99_segmentation.json"
    }
    query = ""

    for name, path in corpus_paths.items():
        print(f"\nProcessing Corpus: {name}")
        corpus = load_corpus(path)
        print(f"Loaded {len(corpus)} chunks.")
        top_chunks = find_top_k(query, corpus, k=30)
        print(f"Top 5 similar chunks from {name}:")
        for idx, (chunk, score) in enumerate(top_chunks, 1):
            print(f"{idx}. Score: {score:.4f}\n   Chunk: {chunk}\n")
            breakpoint()

if __name__ == "__main__":
    main()