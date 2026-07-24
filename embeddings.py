"""
Shared embedding logic used by both the offline indexing script (process_data.py)
and the live API (main.py).

Why a shared module: embeddings produced at index time and at query time MUST come
from the exact same model and produce vectors in the exact same space, or cosine
similarity search is meaningless. Keeping this in one place makes that guarantee
structural instead of "please remember to keep these in sync".

Model choice: all-MiniLM-L6-v2 is a sentence-transformers model trained specifically
for semantic similarity (contrastive training on 1B+ sentence pairs), and it natively
outputs 384-dimensional vectors -- matching the existing Pinecone index dimension with
no reprojection layer required.
"""

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # Loaded lazily and cached: the model is only downloaded/instantiated once
    # per process, on first use, rather than unconditionally at import time.
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts (used when indexing document chunks)."""
    model = _get_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit-normalize so cosine similarity is well-behaved
        show_progress_bar=False,
    )
    return vectors.tolist()


@lru_cache(maxsize=1024)
def embed_query(text: str) -> tuple:
    """
    Embed a single query string (used at request time).

    Cached (by exact question text) since repeated/common questions are cheap to
    recognize and re-embedding is otherwise a synchronous, CPU-bound cost on every
    request. Returns a tuple (hashable, required for lru_cache) -- convert to a
    list with `list(...)` before passing to Pinecone.
    """
    return tuple(embed_texts([text])[0])