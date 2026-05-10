"""Singleton wrapper around SentenceTransformer for embedding generation."""

import os

# Force HuggingFace to use local cache only (avoid network calls that fail on some networks)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

_model = None
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single text string into a 384-dim vector."""
    model = _get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts into 384-dim vectors (batched)."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, batch_size=64).tolist()
