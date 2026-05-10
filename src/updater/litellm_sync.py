"""Fetch pricing data from LiteLLM's GitHub-hosted JSON."""

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

        pricing[model_name] = {
            "input_price_per_mtok": (input_cost or 0) * 1_000_000,
            "output_price_per_mtok": (output_cost or 0) * 1_000_000,
            "context_window": info.get("max_tokens", info.get("max_input_tokens", 0)),
            "max_output_tokens": info.get("max_output_tokens", 0),
            "supports_function_calling": info.get("supports_function_calling", False),
            "supports_vision": info.get("supports_vision", False),
            "litellm_provider": info.get("litellm_provider", ""),
        }

    print(f"  Found pricing for {len(pricing)} models from LiteLLM")
    return pricing


def merge_litellm_pricing(models: list[dict], litellm_data: dict[str, dict]) -> list[dict]:
    """Merge LiteLLM pricing into existing model list where pricing is missing or zero."""
    updated = 0
    for model in models:
        model_id = model.get("id", "")
        model_name = model.get("name", "")

        # Try to find a match in LiteLLM data
        match = None
        for key in [model_id, model_name, model_id.split("/")[-1]]:
            if key in litellm_data:
                match = litellm_data[key]
                break
            # Try with provider prefix variations
            for prefix in ["openai/", "anthropic/", "groq/", "together_ai/", "fireworks_ai/"]:
                if prefix + key in litellm_data:
                    match = litellm_data[prefix + key]
                    break
            if match:
                break

        if match:
            # Fill in missing pricing
            if not model.get("input_price_per_mtok"):
                model["input_price_per_mtok"] = match["input_price_per_mtok"]
                updated += 1
            if not model.get("output_price_per_mtok"):
                model["output_price_per_mtok"] = match["output_price_per_mtok"]
            if not model.get("context_window"):
                model["context_window"] = match["context_window"]

            # Add extra info
            if match.get("supports_function_calling"):
                if "function_calling" not in model.get("tags", []):
                    model.setdefault("tags", []).append("function_calling")
            if match.get("supports_vision"):
                if "vision" not in model.get("tags", []):
                    model.setdefault("tags", []).append("vision")

    print(f"  Merged pricing for {updated} models from LiteLLM")
    return models
