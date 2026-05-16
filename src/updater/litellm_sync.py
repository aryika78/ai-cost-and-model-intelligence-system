"""Fetch pricing data from LiteLLM's GitHub-hosted JSON.

LiteLLM keys follow the pattern: platform/model-name (e.g., "openai/gpt-4o", "groq/llama-3.1-70b").
Keys without a prefix are treated as the implicit canonical platform for that model family.
"""

import requests

LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

# Platforms to exclude — either already covered (openrouter) or have bad data (wandb)
_EXCLUDED_PLATFORMS = {"openrouter", "wandb"}

# OR-specific model variant suffixes that LiteLLM won't have
_OR_SUFFIXES = {":free", ":thinking", ":nitro", ":extended", ":floor", ":online"}


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

    data.pop("sample_spec", None)

    pricing = {}
    skipped = 0
    for model_name, info in data.items():
        if not isinstance(info, dict):
            continue

        # Extract platform from key prefix: "openai/gpt-4o" → "openai"
        platform = ""
        if "/" in model_name:
            platform = model_name.split("/")[0]
        else:
            # No slash — use litellm_provider field as platform (e.g. bedrock-style keys)
            platform = info.get("litellm_provider", "")

        # Skip excluded platforms
        if platform in _EXCLUDED_PLATFORMS:
            skipped += 1
            continue

        input_cost = info.get("input_cost_per_token", 0)
        output_cost = info.get("output_cost_per_token", 0)
        training_cost = info.get("training_cost_per_token", 0)

        pricing[model_name] = {
            "input_price_per_mtok": (input_cost or 0) * 1_000_000,
            "output_price_per_mtok": (output_cost or 0) * 1_000_000,
            "training_cost_per_mtok": (training_cost or 0) * 1_000_000,
            "context_window": info.get("max_tokens", info.get("max_input_tokens", 0)),
            "max_output_tokens": info.get("max_output_tokens", 0),
            "supports_function_calling": info.get("supports_function_calling", False),
            "supports_vision": info.get("supports_vision", False),
            "litellm_provider": info.get("litellm_provider", platform),
            "platform": platform,
            "raw_key": model_name,
        }

    print(f"  Found pricing for {len(pricing)} models from LiteLLM ({skipped} excluded platforms)")
    return pricing


def _strip_or_suffix(s: str) -> str:
    """Strip OpenRouter variant suffixes like :free, :thinking from a model short name."""
    for suffix in _OR_SUFFIXES:
        if s.endswith(suffix):
            return s[: -len(suffix)]
    # Also handle unknown suffixes: anything after ':'
    if ":" in s:
        return s.split(":")[0]
    return s


def _candidate_keys(model_id: str, model_name: str) -> list[str]:
    """Generate candidate LiteLLM key variations to try for a given OR model.

    Handles:
    - OR variant suffixes (:free, :thinking, etc.)
    - Dot vs dash in version numbers (claude-sonnet-4.6 → claude-sonnet-4-6)
    - Date suffixes (claude-3-5-sonnet-20241022 → claude-3-5-sonnet)
    """
    short = model_id.split("/")[-1] if "/" in model_id else model_id

    # Strip OR-specific suffix to get base short name
    short_base = _strip_or_suffix(short)

    # Dots-to-dashes variant (claude-sonnet-4.6 → claude-sonnet-4-6)
    short_base_dash = short_base.replace(".", "-") if "." in short_base else short_base

    candidates = [
        model_id,         # anthropic/claude-sonnet-4.6
        short,            # claude-sonnet-4.6
        short_base,       # claude-sonnet-4.6 (suffix stripped, may equal short)
        short_base_dash,  # claude-sonnet-4-6
        model_name,       # Claude Sonnet 4.6
        model_name.lower(),
    ]

    # Also try stripping 8-digit date suffixes: claude-3-5-sonnet-20241022 → claude-3-5-sonnet
    parts = short_base_dash.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8:
        candidates.append(parts[0])

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def merge_litellm_pricing(models: list[dict], litellm_data: dict[str, dict]) -> list[dict]:
    """Merge LiteLLM pricing into existing model list.

    For each OR model:
    - Fills missing top-level input/output/context if 0
    - Extends pricing[] array with per-platform entries (deduped by platform, case-insensitive)
    - Updates available_platforms[] (deduped, case-insensitive)
    - Fills has_vision / has_function_calling if OR data is False and LiteLLM says True
      (LiteLLM only supplements; OR data takes precedence)
    """
    updated = 0

    for model in models:
        model_id = model.get("id", "")
        model_name = model.get("name", "")

        # --- Phase 1: direct key lookup ---
        matches = []
        tried_keys: set[str] = set()

        for key in _candidate_keys(model_id, model_name):
            if key in tried_keys:
                continue
            tried_keys.add(key)
            if key in litellm_data:
                matches.append(litellm_data[key])

        # --- Phase 2: platform scan — short name comparison with normalization ---
        short = model_id.split("/")[-1] if "/" in model_id else model_id
        short_base = _strip_or_suffix(short)
        short_base_norm = short_base.replace(".", "-").lower()
        name_lower = model_name.lower()

        for key, entry in litellm_data.items():
            if key in tried_keys:
                continue
            key_short = key.split("/")[-1] if "/" in key else key
            key_short_norm = key_short.replace(".", "-").lower()

            if key_short_norm == short_base_norm or key_short.lower() == name_lower:
                matches.append(entry)
                tried_keys.add(key)

        if not matches:
            continue

        updated += 1
        first = matches[0]

        # Fill missing top-level fields from the first match
        if not model.get("input_price_per_mtok"):
            model["input_price_per_mtok"] = first["input_price_per_mtok"]
        if not model.get("output_price_per_mtok"):
            model["output_price_per_mtok"] = first["output_price_per_mtok"]
        if not model.get("context_window"):
            model["context_window"] = first["context_window"]
        if first.get("training_cost_per_mtok", 0) > 0 and not model.get("training_cost_per_mtok"):
            model["training_cost_per_mtok"] = first["training_cost_per_mtok"]

        # Supplement capability flags — only if OR data shows False (not set by OR API)
        # OR API is the primary source; LiteLLM only fills gaps
        if first.get("supports_vision") and not model.get("has_vision"):
            model["has_vision"] = True
        if first.get("supports_function_calling") and not model.get("has_function_calling"):
            model["has_function_calling"] = True

        # Build deduplicated pricing array and available_platforms list
        existing_pricing: list[dict] = model.setdefault("pricing", [])
        existing_platforms_lower = {e.get("platform", "").lower() for e in existing_pricing}

        available_platforms: list[str] = model.setdefault("available_platforms", [])
        available_lower = {p.lower() for p in available_platforms}

        for entry in matches:
            platform = entry.get("litellm_provider") or entry.get("platform") or ""
            if not platform:
                continue
            in_price = entry["input_price_per_mtok"]
            out_price = entry["output_price_per_mtok"]
            if not in_price and not out_price:
                continue

            platform_lower = platform.lower()
            if platform_lower not in existing_platforms_lower:
                existing_pricing.append({
                    "platform": platform,
                    "input_price_per_mtok": in_price,
                    "output_price_per_mtok": out_price,
                })
                existing_platforms_lower.add(platform_lower)

            if platform_lower not in available_lower:
                available_platforms.append(platform)
                available_lower.add(platform_lower)

    print(f"  Merged pricing for {updated} models from LiteLLM")
    return models


def create_new_models_from_litellm(litellm_data: dict[str, dict], existing_models: list[dict]) -> list[dict]:
    """Create new model records for LiteLLM entries not already in our model list.

    Groups per-platform entries into single model records.
    e.g., "groq/llama-3.1-70b", "together_ai/llama-3.1-70b" → one "llama-3.1-70b" record.

    Returns list of new model dicts.
    """
    # Build a set of known short names from existing models (lowercase for dedup)
    known_shorts: set[str] = set()
    for m in existing_models:
        mid = m.get("id", "")
        name = m.get("name", "")
        known_shorts.add(mid.lower())
        known_shorts.add(name.lower())
        if "/" in mid:
            known_shorts.add(mid.split("/")[-1].lower())

    # Group LiteLLM entries by base model name (lowercase to handle case variants)
    groups: dict[str, list[tuple[str, dict]]] = {}

    for key, entry in litellm_data.items():
        platform = entry.get("litellm_provider") or entry.get("platform") or ""
        base = key.split("/")[-1] if "/" in key else key
        base_lower = base.lower()

        if base_lower in known_shorts or key.lower() in known_shorts:
            continue

        if not entry.get("input_price_per_mtok") and not entry.get("output_price_per_mtok"):
            continue

        if base_lower not in groups:
            groups[base_lower] = []
        groups[base_lower].append((platform, entry))

    new_models = []
    for base_name, platform_entries in groups.items():
        if not platform_entries:
            continue

        primary = max(platform_entries, key=lambda x: x[1].get("context_window") or 0)[1]

        pricing = []
        available_platforms = []
        seen_platforms: set[str] = set()

        for platform, entry in platform_entries:
            if not platform:
                continue
            p_lower = platform.lower()
            if entry.get("input_price_per_mtok") or entry.get("output_price_per_mtok"):
                if p_lower not in seen_platforms:
                    pricing.append({
                        "platform": platform,
                        "input_price_per_mtok": entry["input_price_per_mtok"],
                        "output_price_per_mtok": entry["output_price_per_mtok"],
                    })
                    available_platforms.append(platform)
                    seen_platforms.add(p_lower)

        if not pricing:
            continue

        cheapest = min(pricing, key=lambda x: x["input_price_per_mtok"])

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
            "has_vision": bool(primary.get("supports_vision")),
            "has_function_calling": bool(primary.get("supports_function_calling")),
            "has_audio": False,
            "has_image_generation": False,
            "has_reasoning": False,
            "open_source": None,
            "provider": available_platforms[0] if available_platforms else "unknown",
            "source": "litellm",
        })

    print(f"  Created {len(new_models)} new model records from LiteLLM")
    return new_models
