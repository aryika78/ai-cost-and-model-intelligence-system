"""LLM-generated capability profiles for AI models.

Supports two enrichment backends:
- Ollama (default, free/local) — uses qwen3:4b or configured OLLAMA_MODEL
- Groq (fallback, free-tier rate-limited) — set LLM_PROVIDER=groq in .env

The capability profile is embedded into Qdrant alongside model descriptions,
significantly improving semantic search relevance.
"""

import os
import time

GROQ_MODEL = "llama-3.3-70b-versatile"
# Groq free tier: 30 RPM but only 6000 TPM for this model.
# Each call uses ~400 tokens (200 prompt + 200 output).
# 6000 TPM / 400 = 15 calls/min max → 1 call every 4 seconds.
REQUESTS_PER_MINUTE = 15
DELAY_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE  # 4 seconds

ENRICHMENT_PROMPT = """You are an AI model analyst. Given the following AI model info, generate a capability profile.

Model name: {name}
Model ID: {model_id}
Provider: {provider}
Description: {description}
Tags: {tags}
Category: {category}
Context window: {context_window}
Type: {type}

Write a detailed capability profile in 2-3 sentences covering:
1. What tasks/domains this model excels at (be specific: "legal document analysis", "code generation in Python", etc.)
2. What use cases it's best suited for
3. What it should NOT be used for (if obvious from the model type)

Be specific and task-oriented. Do NOT repeat the model name or generic phrases like "general-purpose".
Respond with ONLY the capability profile text, nothing else."""


def _build_prompt(model_data: dict) -> str:
    return ENRICHMENT_PROMPT.format(
        name=model_data.get("name", ""),
        model_id=model_data.get("id", ""),
        provider=model_data.get("provider", ""),
        description=model_data.get("description", "")[:300],
        tags=", ".join(model_data.get("tags", [])),
        category=model_data.get("category", ""),
        context_window=model_data.get("context_window", 0),
        type=model_data.get("type", ""),
    )


def _generate_profile_ollama(model_data: dict) -> str | None:
    """Generate a capability profile using local Ollama (no API cost)."""
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage

        ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
        llm = ChatOllama(model=ollama_model, temperature=0.3)
        prompt = _build_prompt(model_data)
        result = llm.invoke([HumanMessage(content=prompt)])
        text = result.content.strip() if hasattr(result, "content") else str(result).strip()
        # qwen3 sometimes wraps output in <think>...</think> — strip it
        if "<think>" in text and "</think>" in text:
            text = text[text.index("</think>") + len("</think>"):].strip()
        return text if text else None
    except Exception as e:
        print(f"  [WARN] Ollama enrichment failed for '{model_data.get('id')}': {e}")
        return None


def _generate_profile_groq(client, model_data: dict, retries: int = 4) -> str | None:
    """Generate a capability profile using Groq API with retry on rate limit."""
    prompt = _build_prompt(model_data)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower() or "rate limit" in err.lower():
                wait = 60 * (attempt + 1)  # 60s, 120s, 180s, 240s
                print(f"  [RATE LIMIT] Waiting {wait}s before retry {attempt+1}/{retries}...")
                time.sleep(wait)
            else:
                print(f"  [WARN] Groq enrichment failed for '{model_data.get('id')}': {e}")
                return None
    print(f"  [WARN] Gave up on '{model_data.get('id')}' after {retries} retries")
    return None


def enrich_models(models: list[dict]) -> list[dict]:
    """Add capability_profile to each model via LLM calls.

    Uses Ollama by default (free, local). Falls back to Groq if
    LLM_PROVIDER=groq is set and GROQ_API_KEY is available.

    Skips models that already have a capability_profile.
    """
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()

    # Setup client / backend
    use_groq = False
    groq_client = None

    if provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set")
        from groq import Groq
        groq_client = Groq(api_key=api_key)
        use_groq = True
        print(f"  Using Groq ({GROQ_MODEL}) for capability enrichment")
    else:
        ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
        print(f"  Using Ollama ({ollama_model}) for capability enrichment (free/local)")

    to_enrich = [m for m in models if not m.get("capability_profile")]
    already_done = len(models) - len(to_enrich)

    if already_done:
        print(f"  Skipping {already_done} already-enriched models")

    if not to_enrich:
        print("  All models already enriched, nothing to do.")
        return models

    if use_groq:
        est_minutes = len(to_enrich) * DELAY_BETWEEN_REQUESTS / 60
        print(f"  Enriching {len(to_enrich)} models via Groq @ 15/min (est. {est_minutes:.0f} min, ~{est_minutes/60:.1f} hrs)...")
        print(f"  Safe to leave running — retries automatically on rate limit hits.")
    else:
        print(f"  Enriching {len(to_enrich)} models via Ollama (speed depends on CPU)...")

    enriched_count = 0
    failed_count = 0

    for i, model in enumerate(to_enrich):
        if use_groq:
            profile = _generate_profile_groq(groq_client, model)
        else:
            profile = _generate_profile_ollama(model)

        if profile:
            model["capability_profile"] = profile
            enriched_count += 1
        else:
            failed_count += 1

        # Progress update every 10 models
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(to_enrich)} ({enriched_count} enriched, {failed_count} failed)")

        # Rate limiting (only needed for Groq)
        if use_groq and i < len(to_enrich) - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"  Enrichment complete: {enriched_count} enriched, {failed_count} failed")
    return models
