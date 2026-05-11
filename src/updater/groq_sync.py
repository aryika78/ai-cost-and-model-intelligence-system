"""Fetch available models from Groq API.

Requires GROQ_API_KEY in environment. Skips gracefully if not set.
Groq models are fast-inference versions of open-source models (Llama, Mixtral, Gemma, etc.)
"""

import os
import requests


GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"


def fetch_models() -> list[dict]:
    """Fetch available models from Groq. Returns [] if no API key or on failure."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("  [SKIP] groq_sync: GROQ_API_KEY not set — skipping Groq model fetch")
        return []

    print("Fetching models from Groq...")
    try:
        resp = requests.get(
            GROQ_MODELS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [WARN] groq_sync: failed to fetch Groq models: {e}")
        return []

    raw_models = data.get("data", [])
    print(f"  Found {len(raw_models)} models on Groq")

    normalized = []
    for m in raw_models:
        model_id = m.get("id", "")
        if not model_id:
            continue

        normalized.append({
            "id": f"groq/{model_id}",
            "name": model_id,
            "description": f"Available on Groq (fast inference). Context: {m.get('context_window', 0):,} tokens.",
            "context_window": m.get("context_window", 0),
            # Actual per-token pricing filled by LiteLLM merge step in run_update
            # pricing[] starts empty so merge_litellm_pricing can add real Groq prices
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
            "pricing": [],
            "available_platforms": ["groq"],
            # type/category/open_source: None — enricher fills via LLM
            "type": None,
            "category": None,
            "open_source": None,
            "tags": [],
            "provider": "groq",
            "source": "groq",
        })

    return normalized
