"""Orchestrate all data sources, deduplicate, and upsert to Qdrant."""

from dotenv import load_dotenv
load_dotenv()

from src.updater.openrouter_sync import fetch_models as fetch_openrouter
from src.updater.litellm_sync import fetch_pricing as fetch_litellm, merge_litellm_pricing
from src.updater.huggingface_sync import fetch_models as fetch_huggingface
from src.updater.capability_enricher import enrich_models
from src.db.qdrant_manager import create_collection, upsert_models, get_collection_count


def run_update():
    """Run the full data update pipeline."""
    print("=" * 60)
    print("Starting model database update...")
    print("=" * 60)

    # Step 1: Create collection if needed
    create_collection()

    # Step 2: Fetch from all sources
    openrouter_models = fetch_openrouter()
    litellm_pricing = fetch_litellm()
    hf_models = fetch_huggingface()

    # Step 3: Merge LiteLLM pricing into OpenRouter models
    if litellm_pricing:
        openrouter_models = merge_litellm_pricing(openrouter_models, litellm_pricing)

    # Step 4: Deduplicate HF models against OpenRouter models
    or_names = set()
    for m in openrouter_models:
        or_names.add(m["id"].lower())
        or_names.add(m["name"].lower())
        # Also add the short name (after /)
        if "/" in m["id"]:
            or_names.add(m["id"].split("/")[-1].lower())

    unique_hf = []
    for m in hf_models:
        hf_name = m.get("hf_model_id", "").lower()
        short_name = hf_name.split("/")[-1] if "/" in hf_name else hf_name
        # Skip if already in OpenRouter
        if short_name in or_names or hf_name in or_names:
            continue
        unique_hf.append(m)

    print(f"\nDeduplication: {len(hf_models)} HF models -> {len(unique_hf)} unique")

    # Step 5: Combine all models
    all_models = openrouter_models + unique_hf
    print(f"\nTotal models to upsert: {len(all_models)}")

    # Step 6: Enrich models with LLM-generated capability profiles
    print("\nEnriching models with capability profiles (Groq LLM)...")
    try:
        all_models = enrich_models(all_models)
    except Exception as e:
        print(f"  [WARN] Enrichment failed ({e}), continuing with raw descriptions")

    # Step 7: Upsert to Qdrant (in batches)
    if all_models:
        print("Embedding and upserting models (this may take a moment)...")
        upsert_models(all_models)

    # Step 8: Report
    count = get_collection_count()
    print(f"\n{'=' * 60}")
    print(f"Update complete!")
    print(f"  OpenRouter models: {len(openrouter_models)}")
    print(f"  HuggingFace unique models: {len(unique_hf)}")
    print(f"  Total in database: {count}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_update()
