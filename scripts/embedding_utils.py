from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer(MODEL_NAME)


def encode_texts(texts):
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    model = get_embedder()
    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    return embeddings
