"""Orchestrate data pipeline: fetch OpenRouter → merge LiteLLM pricing → upsert to Qdrant."""

from dotenv import load_dotenv
load_dotenv()

from src.updater.openrouter_sync import fetch_models as fetch_openrouter
from src.updater.litellm_sync import fetch_pricing as fetch_litellm, merge_litellm_pricing
from src.updater.gpu_pricing_sync import update_gpu_pricing
from src.db.qdrant_manager import create_collection, wipe_collection, upsert_models, get_collection_count


def run_update():
    """Run the full data update pipeline.

    Pipeline:
      1. Update GPU pricing (SkyPilot Catalog)
      2. Wipe + recreate DB collection (clean slate)
      3. Fetch OpenRouter models (source of truth: 356 models, ~8 alias models excluded)
      4. Fetch LiteLLM pricing (fills per-platform pricing gaps)
      5. Merge LiteLLM pricing into OR models
      6. Upsert everything to Qdrant
    """
    print("=" * 60)
    print("Starting model database update...")
    print("=" * 60)

    # Step 1: Update GPU pricing from SkyPilot Catalog
    print()
    update_gpu_pricing()

    # Step 2: Wipe and recreate collection (guaranteed clean start)
    print()
    wipe_collection()
    create_collection()

    # Step 3: Fetch OpenRouter models
    print()
    or_models = fetch_openrouter()
    if not or_models:
        print("[ERROR] No OpenRouter models fetched — aborting.")
        return

    # Step 4: Fetch LiteLLM pricing
    print()
    litellm_data = fetch_litellm()

    # Step 5: Merge LiteLLM pricing into OR models
    # - Adds per-platform pricing entries (bedrock, azure, groq, etc.)
    # - Fills context_window / pricing gaps where OR has 0
    # - Supplements has_vision / has_function_calling where OR data is missing
    if litellm_data:
        print()
        or_models = merge_litellm_pricing(or_models, litellm_data)

    # Step 6: Upsert all models to Qdrant
    print()
    print(f"Upserting {len(or_models)} models to database...")
    upsert_models(or_models)

    # Step 7: Report
    count = get_collection_count()
    print()
    print("=" * 60)
    print("Update complete!")
    print(f"  OpenRouter models fetched: {len(or_models)}")
    print(f"  Total in database: {count}")

    # Capability summary
    has_vision = sum(1 for m in or_models if m.get("has_vision"))
    has_fc = sum(1 for m in or_models if m.get("has_function_calling"))
    has_reasoning = sum(1 for m in or_models if m.get("has_reasoning"))
    has_audio = sum(1 for m in or_models if m.get("has_audio"))
    has_imggen = sum(1 for m in or_models if m.get("has_image_generation"))
    open_src = sum(1 for m in or_models if m.get("open_source") is True)
    with_pricing = sum(1 for m in or_models if m.get("input_price_per_mtok", 0) > 0)
    multi_platform = sum(1 for m in or_models if len(m.get("available_platforms", [])) > 1)

    print(f"  With pricing:            {with_pricing}")
    print(f"  Multi-platform pricing:  {multi_platform}")
    print(f"  Open source (confirmed): {open_src}")
    print(f"  Vision:                  {has_vision}")
    print(f"  Function calling:        {has_fc}")
    print(f"  Reasoning:               {has_reasoning}")
    print(f"  Audio:                   {has_audio}")
    print(f"  Image generation:        {has_imggen}")
    print("=" * 60)


if __name__ == "__main__":
    run_update()
