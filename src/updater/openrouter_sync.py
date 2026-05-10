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

        # Determine tags from model ID and architecture
        tags = _extract_tags(m)

        normalized.append({
            "id": model_id,
            "name": m.get("name", model_id),
            "description": m.get("description", ""),
            "context_window": m.get("context_length", 0),
            "input_price_per_mtok": input_price * 1_000_000 if input_price else 0,
            "output_price_per_mtok": output_price * 1_000_000 if output_price else 0,
            "type": _infer_type(m),
            "category": _infer_category(model_id, m),
            "tags": tags,
            "open_source": _is_open_source(model_id),
            "provider": _extract_provider(model_id),
            "source": "openrouter",
            "modalities": {
                "input": m.get("architecture", {}).get("modality", "text->text").split("->")[0].split("+"),
                "output": m.get("architecture", {}).get("modality", "text->text").split("->")[-1].split("+"),
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


def _extract_tags(model: dict) -> list[str]:
    """Extract meaningful tags from model data."""
    tags = []
    model_id = model.get("id", "").lower()
    arch = model.get("architecture", {})

    modality = arch.get("modality", "text->text")
    if "image" in modality:
        tags.append("vision")
    if "audio" in modality:
        tags.append("audio")

    if arch.get("instruct_type"):
        tags.append("instruct")

    if "chat" in model_id or "instruct" in model_id:
        tags.append("chat")
    if "code" in model_id:
        tags.append("code")
    if "embed" in model_id:
        tags.append("embedding")
    if "vision" in model_id or "vl" in model_id:
        tags.append("vision")

    return list(set(tags))


def _infer_type(model: dict) -> str:
    """Infer model type from architecture."""
    model_id = model.get("id", "").lower()
    if "embed" in model_id:
        return "embedding"
    if "image" in model_id or "dall-e" in model_id or "stable-diffusion" in model_id or "flux" in model_id:
        return "image_generation"
    if "tts" in model_id or "whisper" in model_id:
        return "audio"
    return "chat"


def _infer_category(model_id: str, model: dict) -> str:
    """Infer model category."""
    mid = model_id.lower()
    if "embed" in mid:
        return "embedding"
    if any(x in mid for x in ["dall-e", "stable-diffusion", "flux", "sdxl"]):
        return "image_generation"
    ctx = model.get("context_length", 0)
    if ctx and ctx >= 100000:
        return "long_context"
    if "code" in mid:
        return "code"
    return "general"


def _is_open_source(model_id: str) -> bool:
    """Heuristic: check if model is open-source based on ID."""
    open_source_families = [
        "llama", "mistral", "mixtral", "qwen", "gemma", "phi", "deepseek",
        "yi", "falcon", "mpt", "command-r", "dbrx", "olmo", "starcoder",
        "codellama", "vicuna", "openchat", "solar", "internlm",
    ]
    mid = model_id.lower()
    return any(f in mid for f in open_source_families)


def _extract_provider(model_id: str) -> str:
    """Extract provider/org from model ID (e.g., 'meta-llama/llama-3' -> 'meta-llama')."""
    if "/" in model_id:
        return model_id.split("/")[0]
    return "unknown"
