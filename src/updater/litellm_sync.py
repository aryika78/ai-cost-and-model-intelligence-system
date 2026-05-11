"""Fetch pricing data from LiteLLM's GitHub-hosted JSON.

LiteLLM keys follow the pattern: platform/model-name (e.g., "openai/gpt-4o", "groq/llama-3.1-70b").
Keys without a prefix are treated as the implicit canonical platform for that model family.
"""

import requests

LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"


def fetch_pricing() -> dict[str, dict]:
    """Fetch pricing data from LiteLLM. Returns dict keyed by model name."""
    print("Fetching pricing from LiteLLM...")
    try:
        resp = requests.get(LITELLM_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Error fetching LiteLLM data: {e}")
        return {}

    # Remove the sample_spec entry
    data.pop("sample_spec", None)

    pricing = {}
    for model_name, info in data.items():
        if not isinstance(info, dict):
            continue
        input_cost = info.get("input_cost_per_token", 0)
        output_cost = info.get("output_cost_per_token", 0)
        training_cost = info.get("training_cost_per_token", 0)

        # Extract platform from key: "openai/gpt-4o" → "openai"
        platform = ""
        if "/" in model_name:
            platform = model_name.split("/")[0]

        pricing[model_name] = {
            "input_price_per_mtok": (input_cost or 0) * 1_000_000,
            "output_price_per_mtok": (output_cost or 0) * 1_000_000,
            "training_cost_per_mtok": (training_cost or 0) * 1_000_000,
            "context_window": info.get("max_tokens", info.get("max_input_tokens", 0)),
            "max_output_tokens": info.get("max_output_tokens", 0),
            "supports_function_calling": info.get("supports_function_calling", False),
            "supports_vision": info.get("supports_vision", False),
            "litellm_provider": info.get("litellm_provider", platform),
            "platform": platform,  # parsed from key prefix
            "raw_key": model_name,
        }

    print(f"  Found pricing for {len(pricing)} models from LiteLLM")
    return pricing


def _candidate_keys(model_id: str, model_name: str) -> list[str]:
    """Generate all possible key variations to try against LiteLLM data."""
    short = model_id.split("/")[-1] if "/" in model_id else model_id
    candidates = [
        model_id,            # openai/gpt-4o
        short,               # gpt-4o  ← LiteLLM usually uses this
        model_name,          # GPT-4o
        model_name.lower(),  # gpt-4o
    ]
    # Also try stripping date suffixes: claude-3-5-sonnet-20241022 → claude-3-5-sonnet
    parts = short.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8:
        candidates.append(parts[0])
    return candidates


def merge_litellm_pricing(models: list[dict], litellm_data: dict[str, dict]) -> list[dict]:
    """Merge LiteLLM pricing into existing model list.

    - Fills missing input/output/context if not set
    - Extends the `pricing` array with per-platform entries from LiteLLM
    - Updates `available_platforms` list
    - Fills training_cost_per_mtok if available
    - Adds capability tags (function_calling, vision)
    """
    updated = 0
    for model in models:
        model_id = model.get("id", "")
        model_name = model.get("name", "")

        # Find all LiteLLM entries that match this model (could be multiple platforms)
        matches = []
        tried_keys = set()
        for key in _candidate_keys(model_id, model_name):
            if key in tried_keys:
                continue
            tried_keys.add(key)
            if key in litellm_data:
                matches.append(litellm_data[key])

        # Also find all platform-specific variants: e.g., "groq/llama-3.1-70b" matching "llama-3.1-70b"
        short = model_id.split("/")[-1] if "/" in model_id else model_id
        for key, entry in litellm_data.items():
            if key in tried_keys:
                continue
            key_short = key.split("/")[-1] if "/" in key else key
            if key_short.lower() == short.lower() or key_short.lower() == model_name.lower():
                matches.append(entry)
                tried_keys.add(key)

        if not matches:
            continue

        updated += 1

        # Use first match for filling missing top-level pricing fields
        first = matches[0]
        if not model.get("input_price_per_mtok"):
            model["input_price_per_mtok"] = first["input_price_per_mtok"]
        if not model.get("output_price_per_mtok"):
            model["output_price_per_mtok"] = first["output_price_per_mtok"]
        if not model.get("context_window"):
            model["context_window"] = first["context_window"]
        if first.get("training_cost_per_mtok", 0) > 0 and not model.get("training_cost_per_mtok"):
            model["training_cost_per_mtok"] = first["training_cost_per_mtok"]

        # Extend pricing array and available_platforms with each platform entry
        existing_pricing = model.setdefault("pricing", [])
        existing_platforms_in_pricing = {e.get("platform", "") for e in existing_pricing}
        available_platforms = model.setdefault("available_platforms", [])

        for entry in matches:
            platform = entry.get("litellm_provider") or entry.get("platform") or ""
            if not platform:
                continue
            in_price = entry["input_price_per_mtok"]
            out_price = entry["output_price_per_mtok"]
            if not in_price and not out_price:
                continue
            if platform not in existing_platforms_in_pricing:
                existing_pricing.append({
                    "platform": platform,
                    "input_price_per_mtok": in_price,
                    "output_price_per_mtok": out_price,
                })
                existing_platforms_in_pricing.add(platform)
            if platform not in available_platforms:
                available_platforms.append(platform)

        # Add capability tags
        if first.get("supports_function_calling"):
            if "function_calling" not in model.get("tags", []):
                model.setdefault("tags", []).append("function_calling")
        if first.get("supports_vision"):
            if "vision" not in model.get("tags", []):
                model.setdefault("tags", []).append("vision")

    print(f"  Merged pricing for {updated} models from LiteLLM")
    return models


def create_new_models_from_litellm(litellm_data: dict[str, dict], existing_models: list[dict]) -> list[dict]:
    """Create new model records for LiteLLM entries not already in our model list.

    Groups per-platform entries into single model records.
    e.g., "groq/llama-3.1-70b", "together_ai/llama-3.1-70b" → one "llama-3.1-70b" model record.

    Returns list of new model dicts (type/category=None, enricher will fill).
    """
    # Build a set of known short names from existing models
    known_shorts = set()
    for m in existing_models:
        mid = m.get("id", "")
        name = m.get("name", "")
        known_shorts.add(mid.lower())
        known_shorts.add(name.lower())
        if "/" in mid:
            known_shorts.add(mid.split("/")[-1].lower())

    # Group LiteLLM entries by base model name
    # base_name → list of (platform, entry)
    groups: dict[str, list[tuple[str, dict]]] = {}

    for key, entry in litellm_data.items():
        platform = entry.get("litellm_provider") or entry.get("platform") or ""
        base = key.split("/")[-1] if "/" in key else key

        # Skip if already in our model list
        if base.lower() in known_shorts or key.lower() in known_shorts:
            continue

        # Skip entries with no pricing (not useful as standalone models)
        if not entry.get("input_price_per_mtok") and not entry.get("output_price_per_mtok"):
            continue

        if base not in groups:
            groups[base] = []
        groups[base].append((platform, entry))

    new_models = []
    for base_name, platform_entries in groups.items():
        if not platform_entries:
            continue

        # Use the entry with the most context as the "primary" for top-level fields
        primary = max(platform_entries, key=lambda x: x[1].get("context_window") or 0)[1]

        pricing = []
        available_platforms = []
        for platform, entry in platform_entries:
            if platform and (entry.get("input_price_per_mtok") or entry.get("output_price_per_mtok")):
                pricing.append({
                    "platform": platform,
                    "input_price_per_mtok": entry["input_price_per_mtok"],
                    "output_price_per_mtok": entry["output_price_per_mtok"],
                })
                if platform not in available_platforms:
                    available_platforms.append(platform)

        if not pricing:
            continue

        # Cheapest price across platforms as the top-level (for DB range filters)
        cheapest = min(pricing, key=lambda x: x["input_price_per_mtok"])

        tags = []
        if primary.get("supports_function_calling"):
            tags.append("function_calling")
        if primary.get("supports_vision"):
            tags.append("vision")

        new_models.append({
            "id": f"litellm/{base_name}",
            "name": base_name,
            "description": f"Model available via {', '.join(available_platforms)}.",
            "context_window": primary.get("context_window") or 0,
            "input_price_per_mtok": cheapest["input_price_per_mtok"],
            "output_price_per_mtok": cheapest["output_price_per_mtok"],
            "training_cost_per_mtok": primary.get("training_cost_per_mtok") or 0,
            "pricing": pricing,
            "available_platforms": available_platforms,
            # type/category/open_source: None — enricher fills via LLM
            "type": None,
            "category": None,
            "open_source": None,
            "tags": tags,
            "provider": available_platforms[0] if available_platforms else "unknown",
            "source": "litellm",
        })

    print(f"  Created {len(new_models)} new model records from LiteLLM (not in OpenRouter/HF)")
    return new_models
