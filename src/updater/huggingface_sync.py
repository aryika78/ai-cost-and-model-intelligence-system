"""Fetch open-source model data from HuggingFace Hub API."""

import json
import time
import ssl
import urllib.request
import urllib.parse

HF_API_URL = "https://huggingface.co/api/models"

_ssl_ctx = ssl.create_default_context()

# Framework tags — factual metadata from HF, not capability inferences
FRAMEWORK_TAGS = {"pytorch", "transformers", "gguf", "safetensors", "jax", "tensorflow"}


def _fetch_url(url: str, max_retries: int = 5) -> bytes:
    """Fetch URL with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=30, context=_ssl_ctx)
            return resp.read()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    Retry {attempt + 1}/{max_retries} after {wait}s ({e})")
                time.sleep(wait)
            else:
                raise


# Fetch across these pipeline categories to get broad coverage
HF_PIPELINE_CATEGORIES = [
    "text-generation",
    "feature-extraction",
    "text-to-image",
    "automatic-speech-recognition",
    "image-text-to-text",
]


def fetch_models(limit: int = 200) -> list[dict]:
    """Fetch popular open-source models from HuggingFace."""
    print("Fetching models from HuggingFace...")

    all_models = []
    for tag in HF_PIPELINE_CATEGORIES:
        try:
            params = urllib.parse.urlencode({
                "pipeline_tag": tag,
                "sort": "downloads",
                "direction": "-1",
                "limit": limit // len(HF_PIPELINE_CATEGORIES),
            })
            url = f"{HF_API_URL}?{params}"
            data = _fetch_url(url)
            models = json.loads(data.decode())
            all_models.extend(models)
            print(f"  Fetched {len(models)} models for {tag}")
        except Exception as e:
            print(f"  Error fetching HF models for {tag} (after retries): {e}")
        time.sleep(1)

    print(f"  Found {len(all_models)} models from HuggingFace")

    normalized = []
    seen_ids = set()
    for m in all_models:
        model_id = m.get("modelId", m.get("id", ""))
        if not model_id or model_id in seen_ids:
            continue
        seen_ids.add(model_id)

        downloads = m.get("downloads", 0)
        if downloads < 1000:
            continue

        pipeline_tag = m.get("pipeline_tag", "")

        # Only keep factual framework tags — no capability inference from tags
        framework_tags = [t for t in m.get("tags", []) if t in FRAMEWORK_TAGS]

        normalized.append({
            "id": f"hf/{model_id}",
            "name": model_id.split("/")[-1] if "/" in model_id else model_id,
            "description": f"Open-source model on HuggingFace. Pipeline: {pipeline_tag}. "
                           f"Downloads: {downloads:,}. Likes: {m.get('likes', 0)}.",
            "context_window": 0,
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
            # type/category intentionally None — enricher fills via LLM
            "type": None,
            "category": None,
            # open_source is factual for ALL HuggingFace models
            "open_source": True,
            # pipeline_tag stored raw — enricher uses it as context, no mapping
            "pipeline_tag": pipeline_tag,
            "tags": framework_tags,
            "provider": model_id.split("/")[0] if "/" in model_id else "community",
            "source": "huggingface",
            "downloads": downloads,
            "likes": m.get("likes", 0),
            "parameter_count": _estimate_params(model_id),
            "hf_model_id": model_id,
        })

    return normalized


def _estimate_params(model_id: str) -> float | None:
    """Extract parameter count in billions from model ID string.

    Returns a float (e.g. 7.0, 70.0, 0.5) or None if not found.
    Works for any size — 3b, 8b, 14b, 30b, 72b, 235b, 500m, etc.
    """
    import re
    mid = model_id.lower()
    match = re.search(r'(\d+\.?\d*)(b|m)(?:\b|-)', mid)
    if match:
        num = float(match.group(1))
        unit = match.group(2)
        return num if unit == "b" else round(num / 1000, 3)
    return None
