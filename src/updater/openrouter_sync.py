"""Fetch AI model data from OpenRouter API (no auth needed)."""

import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"


def fetch_models() -> list[dict]:
    """Fetch all models from OpenRouter and normalize to our schema."""
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
    for m in raw_models:
        model_id = m.get("id", "")
        if not model_id:
            continue

        pricing = m.get("pricing", {})
        input_price = _parse_price(pricing.get("prompt", "0"))
        output_price = _parse_price(pricing.get("completion", "0"))

        arch = m.get("architecture", {})

        input_price_mtok = input_price * 1_000_000 if input_price else 0
        output_price_mtok = output_price * 1_000_000 if output_price else 0

        normalized.append({
            "id": model_id,
            "name": m.get("name", model_id),
            "description": m.get("description", ""),
            "context_window": m.get("context_length", 0),
            "input_price_per_mtok": input_price_mtok,
            "output_price_per_mtok": output_price_mtok,
            # Per-platform pricing array (extended by litellm_sync with other platforms)
            # Always add openrouter — even free ($0) models are available on the platform
            "pricing": [
                {
                    "platform": "openrouter",
                    "input_price_per_mtok": input_price_mtok,
                    "output_price_per_mtok": output_price_mtok,
                }
            ] if input_price_mtok or output_price_mtok else [],
            "available_platforms": ["openrouter"],
            # type/category/open_source/tags: None — enricher assigns via LLM
            "type": None,
            "category": None,
            "open_source": None,
            "tags": [],
            "provider": _extract_provider(model_id),
            "source": "openrouter",
            # Store exactly what the API gives — no parsing or inference
            "modalities": {
                "input": arch.get("input_modalities", ["text"]),
                "output": arch.get("output_modalities", ["text"]),
            },
            "top_provider": m.get("top_provider", {}),
        })

    return normalized


def _parse_price(price_str) -> float:
    """Parse price string to float (price per token)."""
    try:
        return float(price_str)
    except (ValueError, TypeError):
        return 0.0


def _extract_provider(model_id: str) -> str:
    """Extract provider/org from model ID (e.g., 'meta-llama/llama-3' -> 'meta-llama')."""
    if "/" in model_id:
        return model_id.split("/")[0]
    return "unknown"
