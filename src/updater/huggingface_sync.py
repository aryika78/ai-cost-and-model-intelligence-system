"""Fetch open-source model data from HuggingFace Hub API."""

import json
import time
import ssl
import urllib.request
import urllib.parse

HF_API_URL = "https://huggingface.co/api/models"

# Create a custom SSL context that's more lenient with connections
_ssl_ctx = ssl.create_default_context()


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

RELEVANT_PIPELINE_TAGS = [
    "text-generation",
    "text2text-generation",
    "text-classification",
    "token-classification",
    "question-answering",
    "summarization",
    "translation",
    "feature-extraction",
    "sentence-similarity",
    "image-text-to-text",
    "image-classification",
    "object-detection",
    "automatic-speech-recognition",
    "text-to-speech",
    "text-to-image",
]


def fetch_models(limit: int = 200) -> list[dict]:
    """Fetch popular open-source models from HuggingFace."""
    print("Fetching models from HuggingFace...")

    all_models = []
    for tag in ["text-generation", "feature-extraction", "text-to-image",
                "automatic-speech-recognition", "image-text-to-text"]:
        try:
            params = urllib.parse.urlencode({
                "pipeline_tag": tag,
                "sort": "downloads",
                "direction": "-1",
                "limit": limit // 5,
            })
            url = f"{HF_API_URL}?{params}"
            data = _fetch_url(url)
            models = json.loads(data.decode())
            all_models.extend(models)
            print(f"  Fetched {len(models)} models for {tag}")
        except Exception as e:
            print(f"  Error fetching HF models for {tag} (after retries): {e}")
        # Small delay between categories to avoid connection resets
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
        tags = m.get("tags", [])
        hf_tags = [t for t in tags if t in RELEVANT_PIPELINE_TAGS or t in [
            "pytorch", "transformers", "gguf", "safetensors"
        ]]

        # Estimate parameter count from model ID
        param_count = _estimate_params(model_id)

        normalized.append({
            "id": f"hf/{model_id}",
            "name": model_id.split("/")[-1] if "/" in model_id else model_id,
            "description": f"Open-source model on HuggingFace. Pipeline: {pipeline_tag}. "
                          f"Downloads: {downloads:,}. Likes: {m.get('likes', 0)}.",
            "context_window": 0,  # Not available from HF API directly
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
            "type": _pipeline_to_type(pipeline_tag),
            "category": _pipeline_to_category(pipeline_tag),
            "tags": hf_tags + [pipeline_tag] if pipeline_tag else hf_tags,
            "open_source": True,
            "provider": model_id.split("/")[0] if "/" in model_id else "community",
            "source": "huggingface",
            "downloads": downloads,
            "likes": m.get("likes", 0),
            "parameter_count": param_count,
            "hf_model_id": model_id,
        })

    return normalized


def _estimate_params(model_id: str) -> str:
    """Estimate parameter count from model ID string."""
    mid = model_id.lower()
    for marker in ["405b", "400b", "340b", "180b", "140b", "120b",
                    "72b", "70b", "65b", "34b", "33b", "27b", "22b",
                    "13b", "14b", "12b", "11b", "9b", "8b", "7b",
                    "3b", "2b", "1.5b", "1b", "0.5b", "500m", "350m", "125m"]:
        if marker in mid:
            return marker
    return "unknown"


def _pipeline_to_type(tag: str) -> str:
    mapping = {
        "text-generation": "chat",
        "text2text-generation": "chat",
        "feature-extraction": "embedding",
        "sentence-similarity": "embedding",
        "text-to-image": "image_generation",
        "automatic-speech-recognition": "audio",
        "text-to-speech": "audio",
        "image-text-to-text": "chat",
        "text-classification": "classification",
        "token-classification": "classification",
        "question-answering": "chat",
        "summarization": "chat",
        "translation": "chat",
    }
    return mapping.get(tag, "other")


def _pipeline_to_category(tag: str) -> str:
    mapping = {
        "text-generation": "general",
        "text2text-generation": "general",
        "feature-extraction": "embedding",
        "sentence-similarity": "embedding",
        "text-to-image": "image_generation",
        "automatic-speech-recognition": "speech",
        "text-to-speech": "speech",
        "image-text-to-text": "multimodal",
    }
    return mapping.get(tag, "specialized")
