"""Fetch AI model data from OpenRouter API (no auth needed)."""

import re
import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"


def fetch_models() -> list[dict]:
    """Fetch all models from OpenRouter and normalize to our schema.

    Excludes ~ prefix alias/redirect models (8 models that are just pointers to others).
    Derives capability flags directly from API modalities + supported_parameters — no LLM.
    """
    print("Fetching models from OpenRouter...")
    try:
        resp = requests.get(OPENROUTER_API_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Error fetching OpenRouter: {e}")
        return []

    raw_models = data.get("data", [])
    print(f"  Found {len(raw_models)} models on OpenRouter")

    normalized = []
    skipped_tilde = 0

    for m in raw_models:
        model_id = m.get("id", "")
        if not model_id:
            continue

        # Skip ~ prefix alias/redirect models (e.g. ~anthropic/claude-haiku-latest)
        # These are just redirects to versioned models — no unique data.
        if "~" in model_id:
            skipped_tilde += 1
            continue

        pricing = m.get("pricing", {})
        input_price = _parse_price(pricing.get("prompt", "0"))
        output_price = _parse_price(pricing.get("completion", "0"))
        input_price_mtok = input_price * 1_000_000 if input_price else 0
        output_price_mtok = output_price * 1_000_000 if output_price else 0

        arch = m.get("architecture", {})
        top_provider = m.get("top_provider", {})
        hf_id = m.get("hugging_face_id") or None

        # open_source: True if hf_id present (confirmed open-source via HF Hub sampling),
        # None if unknown (cannot verify without hf_id).
        open_source = True if hf_id else None

        # Capability flags — derived directly from API data, no keywords or inference
        input_mods = arch.get("input_modalities", ["text"])
        output_mods = arch.get("output_modalities", ["text"])
        supported_params = m.get("supported_parameters", [])

        normalized.append({
            "id": model_id,
            "name": m.get("name", model_id),
            "description": m.get("description", ""),
            "context_window": m.get("context_length", 0),
            "max_output_tokens": top_provider.get("max_completion_tokens") or 0,
            "input_price_per_mtok": input_price_mtok,
            "output_price_per_mtok": output_price_mtok,
            "pricing": [
                {
                    "platform": "openrouter",
                    "input_price_per_mtok": input_price_mtok,
                    "output_price_per_mtok": output_price_mtok,
                }
            ] if input_price_mtok or output_price_mtok else [],
            "available_platforms": ["openrouter"],
            "modalities": {
                "input": input_mods,
                "output": output_mods,
            },
            # Flat capability booleans — derived from modalities + supported_parameters
            # False means "not confirmed" (could be API gap) — do NOT treat as "definitely cannot"
            "has_vision": "image" in input_mods,
            "has_audio": "audio" in input_mods or "audio" in output_mods,
            "has_image_generation": "image" in output_mods,
            "has_function_calling": "tools" in supported_params,
            "has_reasoning": "reasoning" in supported_params,
            "supported_parameters": supported_params,
            # open_source: True (confirmed) or None (unknown)
            "open_source": open_source,
            "hugging_face_id": hf_id,
            "knowledge_cutoff": m.get("knowledge_cutoff") or None,
            "provider": _extract_provider(model_id),
            "source": "openrouter",
            # Param count in billions extracted from model name via regex — None if not found
            "param_count": _extract_param_count(model_id),
        })

    print(f"  Skipped {skipped_tilde} alias (~) models")
    print(f"  Normalized {len(normalized)} OpenRouter models")
    return normalized


def _parse_price(price_str) -> float:
    try:
        return float(price_str)
    except (ValueError, TypeError):
        return 0.0


def _extract_provider(model_id: str) -> str:
    if "/" in model_id:
        return model_id.split("/")[0]
    return "unknown"


def _extract_param_count(model_id: str) -> float | None:
    """Extract parameter count in billions from model ID string.

    Returns float (e.g. 7.0, 70.0, 0.5) or None if not parseable.
    Handles: 7b→7.0, 70b→70.0, 500m→0.5, 1.5b→1.5, 235b→235.0
    Strips OR-specific suffixes (:free, :thinking) before parsing.
    """
    clean_id = model_id.split(":")[0] if ":" in model_id else model_id
    mid = clean_id.lower()
    match = re.search(r'(\d+\.?\d*)(b|m)(?:\b|-)', mid)
    if match:
        num = float(match.group(1))
        unit = match.group(2)
        return num if unit == "b" else round(num / 1000, 3)
    return None
