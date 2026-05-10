"""Qdrant vector database manager for model storage and semantic search."""

import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
)

from src.db.embeddings import embed_text, embed_texts, EMBEDDING_DIM

COLLECTION_NAME = "models"
DATA_PATH = "./data"

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=DATA_PATH)
    return _client


def create_collection():
    """Create the models collection if it doesn't exist."""
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}'")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists")


def _model_id_to_point_id(model_id: str) -> str:
    """Convert a model ID string to a deterministic UUID-like hash for Qdrant."""
    h = hashlib.md5(model_id.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _build_embed_text(model_data: dict) -> str:
    """Build the text to embed from model data fields.

    Prefers LLM-generated capability_profile over raw description for
    much better semantic search relevance.
    """
    capability_profile = model_data.get("capability_profile", "")

    if capability_profile:
        parts = [
            model_data.get("name", ""),
            capability_profile,
            " ".join(model_data.get("tags", [])),
        ]
    else:
        parts = [
            model_data.get("name", ""),
            model_data.get("description", ""),
            " ".join(model_data.get("tags", [])),
            model_data.get("category", ""),
        ]
    return " ".join(p for p in parts if p)


def upsert_model(model_data: dict):
    """Upsert a single model into Qdrant."""
    client = get_client()
    model_id = model_data["id"]
    point_id = _model_id_to_point_id(model_id)
    text = _build_embed_text(model_data)
    vector = embed_text(text)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload=model_data,
            )
        ],
    )


def upsert_models(models: list[dict]):
    """Batch upsert models into Qdrant."""
    if not models:
        return

    client = get_client()
    texts = [_build_embed_text(m) for m in models]
    vectors = embed_texts(texts)

    points = []
    for model_data, vector in zip(models, vectors):
        point_id = _model_id_to_point_id(model_data["id"])
        points.append(
            PointStruct(id=point_id, vector=vector, payload=model_data)
        )

    # Upsert in batches of 100
    for i in range(0, len(points), 100):
        batch = points[i : i + 100]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)


def _build_filter(filters: dict) -> Filter | None:
    """Build Qdrant filter from a dict of field conditions.

    Supported filter keys:
      - type: str (e.g. "chat", "embedding", "image")
      - category: str
      - tags: list[str] (match any)
      - open_source: bool
      - min_context_window: int
      - max_input_price: float (per 1M tokens)
      - provider: str
    """
    if not filters:
        return None

    conditions = []

    if "type" in filters:
        conditions.append(FieldCondition(key="type", match=MatchValue(value=filters["type"])))

    if "category" in filters:
        conditions.append(FieldCondition(key="category", match=MatchValue(value=filters["category"])))

    if "tags" in filters:
        tags = filters["tags"] if isinstance(filters["tags"], list) else [filters["tags"]]
        conditions.append(FieldCondition(key="tags", match=MatchAny(any=tags)))

    if "open_source" in filters:
        conditions.append(FieldCondition(key="open_source", match=MatchValue(value=filters["open_source"])))

    if "min_context_window" in filters:
        conditions.append(
            FieldCondition(key="context_window", range=Range(gte=filters["min_context_window"]))
        )

    if "max_input_price" in filters:
        conditions.append(
            FieldCondition(key="input_price_per_mtok", range=Range(lte=filters["max_input_price"]))
        )

    if "provider" in filters:
        conditions.append(FieldCondition(key="provider", match=MatchValue(value=filters["provider"])))

    if not conditions:
        return None

    return Filter(must=conditions)


def semantic_search(query: str, filters: dict | None = None, top_k: int = 10) -> list[dict]:
    """Semantic search for models matching query and optional metadata filters."""
    client = get_client()
    query_vector = embed_text(query)
    qdrant_filter = _build_filter(filters or {})

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        {**hit.payload, "_score": hit.score}
        for hit in response.points
    ]


def get_model(model_id: str) -> dict | None:
    """Get a single model by its ID."""
    client = get_client()
    point_id = _model_id_to_point_id(model_id)
    try:
        results = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        if results:
            return results[0].payload
    except Exception:
        pass
    return None


def get_models(model_ids: list[str]) -> list[dict]:
    """Get multiple models by their IDs."""
    results = []
    for mid in model_ids:
        m = get_model(mid)
        if m:
            results.append(m)
    return results


def count_results(query: str, filters: dict | None = None) -> int:
    """Count models matching query and filters."""
    results = semantic_search(query, filters, top_k=100)
    # Count those above a relevance threshold
    return len([r for r in results if r.get("_score", 0) > 0.3])


def get_all_tags() -> list[str]:
    """Get all unique tags across all models."""
    client = get_client()
    tags = set()
    offset = None
    while True:
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        points, offset = results
        for point in points:
            for tag in point.payload.get("tags", []):
                tags.add(tag)
        if offset is None:
            break
    return sorted(tags)


def get_collection_count() -> int:
    """Get total number of models in the collection."""
    client = get_client()
    try:
        info = client.get_collection(COLLECTION_NAME)
        return info.points_count
    except Exception:
        return 0
