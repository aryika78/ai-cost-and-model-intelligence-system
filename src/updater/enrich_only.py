"""Standalone enrichment script — enriches unenriched models already in Qdrant.

Use this instead of run_update when you just want to continue enrichment
without re-fetching all models from APIs.

Progress is saved every 10 models — safe to interrupt and resume at any time.
Re-running will skip already-enriched models and pick up where it left off.

Run: python -m src.updater.enrich_only
"""

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from src.updater.capability_enricher import enrich_models
from src.db.qdrant_manager import upsert_models


def enrich_only():
    client = QdrantClient(path="./data")

    # Load all models from Qdrant
    print("Loading all models from database...")
    all_models = []
    offset = None
    while True:
        results = client.scroll(collection_name="models", limit=100, offset=offset, with_payload=True)
        points, offset = results
        for p in points:
            all_models.append(dict(p.payload))
        if offset is None:
            break

    client.close()

    with_profile = sum(1 for m in all_models if m.get("capability_profile"))
    without_profile = len(all_models) - with_profile
    print(f"Total: {len(all_models)} | Already enriched: {with_profile} | Remaining: {without_profile}")

    if without_profile == 0:
        print("All models already enriched!")
        return

    # Enrich only the ones missing profiles.
    # save_callback=upsert_models saves progress every 10 models to DB —
    # so if Groq rate limit kills the process, all completed work is preserved.
    # Re-run this script to resume from where it left off.
    enrich_models(all_models, save_callback=upsert_models, save_every=10)
    print("Done!")


if __name__ == "__main__":
    enrich_only()
