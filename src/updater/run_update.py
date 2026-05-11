"""Orchestrate all data sources, deduplicate, and upsert to Qdrant."""

from dotenv import load_dotenv
load_dotenv()

from src.updater.openrouter_sync import fetch_models as fetch_openrouter
from src.updater.litellm_sync import (
    fetch_pricing as fetch_litellm,
    merge_litellm_pricing,
    create_new_models_from_litellm,
)
from src.updater.huggingface_sync import fetch_models as fetch_huggingface
from src.updater.groq_sync import fetch_models as fetch_groq
from src.updater.capability_enricher import enrich_models
from src.updater.gpu_pricing_sync import update_gpu_pricing
from src.db.qdrant_manager import create_collection, upsert_models, get_collection_count


def _modality_str(model: dict) -> str:
    """Stable string representation of modality for change detection."""
    m = model.get("modalities", {})
    if not m:
        return ""
    inputs = sorted(m.get("input", []))
    outputs = sorted(m.get("output", []))
    return f"{'+'.join(inputs)}->{'+'.join(outputs)}"


def _load_existing_data() -> dict:
    """Load existing profiles + modality from Qdrant to detect what needs re-enrichment."""
    from qdrant_client import QdrantClient
    existing = {}  # id -> {profile, modality_str}
    try:
        client = QdrantClient(path="./data")
        offset = None
        while True:
            results = client.scroll(collection_name="models", limit=100, offset=offset, with_payload=True)
            points, offset = results
            for p in points:
                mid = p.payload.get("id", "")
                if mid:
                    existing[mid] = {
                        "profile": p.payload.get("capability_profile", ""),
                        "modality_str": _modality_str(p.payload),
                    }
            if offset is None:
                break
        client.close()
        with_profile = sum(1 for v in existing.values() if v["profile"])
        print(f"  Loaded {len(existing)} existing models ({with_profile} enriched)")
    except Exception as e:
        print(f"  [WARN] Could not load existing data: {e}")
    return existing


def _dedup_by_short_name(models: list[dict], known_names: set) -> list[dict]:
    """Filter out models whose short name (after last /) already appears in known_names."""
    unique = []
    for m in models:
        mid = m.get("id", "")
        name = m.get("name", "").lower()
        short = mid.split("/")[-1].lower() if "/" in mid else mid.lower()
        if short in known_names or name in known_names or mid.lower() in known_names:
            continue
        unique.append(m)
    return unique


def run_update():
    """Run the full data update pipeline."""
    print("=" * 60)
    print("Starting model database update...")
    print("=" * 60)

    # Step 0: Update GPU pricing from SkyPilot Catalog
    update_gpu_pricing()

    # Step 1: Create collection if needed
    create_collection()

    # Step 1b: Load existing data BEFORE fetching — never lose profiles
    print("\nLoading existing database state...")
    existing_data = _load_existing_data()

    # Step 2: Fetch from all sources
    print()
    openrouter_models = fetch_openrouter()
    litellm_pricing = fetch_litellm()
    hf_models = fetch_huggingface()
    groq_models = fetch_groq()

    # Step 3: Merge LiteLLM pricing into OpenRouter models
    # Extends pricing[] array and available_platforms[] with per-platform entries
    if litellm_pricing:
        openrouter_models = merge_litellm_pricing(openrouter_models, litellm_pricing)

    # Step 4: Create new model records from LiteLLM for models not in OpenRouter/HF
    combined_for_dedup = openrouter_models + hf_models
    litellm_new_models = []
    if litellm_pricing:
        litellm_new_models = create_new_models_from_litellm(litellm_pricing, combined_for_dedup)

    # Step 5: Build known name set from OpenRouter for deduplication
    or_names: set[str] = set()
    for m in openrouter_models:
        mid = m["id"].lower()
        or_names.add(mid)
        or_names.add(m["name"].lower())
        if "/" in mid:
            or_names.add(mid.split("/")[-1])

    # Step 5b: Deduplicate HF models against OpenRouter
    unique_hf = _dedup_by_short_name(hf_models, or_names)
    print(f"\nDeduplication: {len(hf_models)} HF models -> {len(unique_hf)} unique")

    # Step 5c: Deduplicate Groq models against OpenRouter + HF
    all_known = set(or_names)
    for m in unique_hf:
        all_known.add(m.get("name", "").lower())
        all_known.add(m.get("id", "").lower())
    unique_groq = _dedup_by_short_name(groq_models, all_known)
    print(f"Deduplication: {len(groq_models)} Groq models -> {len(unique_groq)} unique")

    # Step 5c.1: Merge LiteLLM pricing into Groq-unique models (they start with $0 pricing)
    if litellm_pricing and unique_groq:
        unique_groq = merge_litellm_pricing(unique_groq, litellm_pricing)

    # Step 5d: Merge Groq pricing into OpenRouter models where available
    # (Groq may serve the same model; add groq as a platform entry if model matches)
    if litellm_pricing:
        groq_litellm = {k: v for k, v in litellm_pricing.items()
                        if v.get("litellm_provider") == "groq" or k.startswith("groq/")}
        if groq_litellm:
            openrouter_models = merge_litellm_pricing(openrouter_models, groq_litellm)

    # Step 6: Combine all models
    all_models = openrouter_models + unique_hf + unique_groq + litellm_new_models
    print(f"\nTotal models to process: {len(all_models)}")

    # Step 7: Smart profile handling
    # - New model                  → no profile → enrich
    # - Existing, modality same    → restore profile → skip enrichment
    # - Existing, modality changed → clear profile → re-enrich
    new_count = restored = reenrich_count = 0
    for m in all_models:
        mid = m.get("id", "")
        if mid not in existing_data:
            new_count += 1
        else:
            old = existing_data[mid]
            old_profile = old.get("profile", "")
            old_modality = old.get("modality_str", "")
            new_modality = _modality_str(m)

            if old_profile and old_modality == new_modality:
                m["capability_profile"] = old_profile
                restored += 1
            elif old_profile and old_modality != new_modality and new_modality:
                print(f"  [RE-ENRICH] {mid}: modality {old_modality} → {new_modality}")
                reenrich_count += 1

    print(f"  New: {new_count} | Profiles kept: {restored} | Re-enriching: {reenrich_count}")

    # Step 8: Upsert ALL models to DB now (without waiting for enrichment).
    # This means even if enrichment is interrupted, the base model data is safe.
    if all_models:
        print("\nUpserting base models to database (before enrichment)...")
        upsert_models(all_models)

    # Step 9: Enrich incrementally — saves every 10 models to DB.
    # If Groq rate limit kills the process mid-way, all completed profiles are saved.
    # Re-run `python -m src.updater.enrich_only` to resume enrichment without re-fetching.
    print("\nEnriching models with capability profiles (LLM)...")
    try:
        enrich_models(all_models, save_callback=upsert_models, save_every=10)
    except Exception as e:
        print(f"  [WARN] Enrichment interrupted ({e}). Models are in DB — run enrich_only to resume.")

    # Step 10: Report
    count = get_collection_count()
    print(f"\n{'=' * 60}")
    print(f"Update complete!")
    print(f"  OpenRouter models: {len(openrouter_models)}")
    print(f"  HuggingFace unique: {len(unique_hf)}")
    print(f"  Groq unique: {len(unique_groq)}")
    print(f"  LiteLLM new: {len(litellm_new_models)}")
    print(f"  Total in database: {count}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_update()
