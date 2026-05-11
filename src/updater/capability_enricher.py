"""LLM-generated capability profiles for AI models.

Supports two enrichment backends:
- Groq (recommended) — set LLM_PROVIDER=groq in .env
- Ollama (local fallback) — set LLM_PROVIDER=ollama

The enricher does two things in one LLM call:
1. Generates a capability profile (for semantic search)
2. Assigns type, category, tags, open_source (since syncs no longer hardcode these)
"""

import os
import time
import json


class _DailyLimitReached(Exception):
    """Groq daily token quota exhausted — no point retrying until tomorrow."""
    pass


GROQ_MODEL = "llama-3.3-70b-versatile"
# Groq free tier: 30 RPM, 6000 TPM.
# Each call uses ~500 tokens (prompt ~300 + output ~200).
# 6000 / 500 = 12 calls/min safe → 1 call every 5 seconds.
REQUESTS_PER_MINUTE = 12
DELAY_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE  # 5 seconds

ENRICHMENT_PROMPT = """You are an AI model analyst. Analyze this AI model and return a JSON object.

Model info:
- Name: {name}
- ID: {model_id}
- Provider: {provider}
- Description: {description}
- Pipeline tag (HuggingFace): {pipeline_tag}
- Input modalities: {input_modalities}
- Output modalities: {output_modalities}
- Context window: {context_window} tokens
- Parameter count: {parameter_count}
- Source: {source}

Return ONLY a valid JSON object with exactly these fields:

{{
  "capability_profile": "2-3 sentences: what tasks this model excels at (be specific: legal analysis, Python code generation, etc.), what use cases it suits, and what it should NOT be used for if obvious.",
  "type": "one of: chat | embedding | image_generation | audio | multimodal | code | reranking | other",
  "category": "one of: general | code | embedding | image_generation | speech | multimodal | long_context | reasoning | specialized",
  "tags": ["list", "of", "relevant", "capability", "tags"],
  "open_source": true or false
}}

Rules:
- capability_profile: be task-specific. Never say "general-purpose". Mention actual domains.
- type: use "chat" for text generation/conversation, "embedding" for vector embeddings, "multimodal" only if both input AND output are multi-modal
- category: use "long_context" if context window >= 100000, "reasoning" if model is known for chain-of-thought or math
- tags: include relevant ones from: vision, audio, code, instruct, multilingual, long-context, function-calling, reasoning, math, medical, legal, embedding, image-generation, speech-recognition, text-to-speech, file-input
- open_source: true if weights are publicly available (HuggingFace models, Llama, Mistral, Qwen, Gemma, Phi, DeepSeek, Falcon, etc.), false for closed models (GPT, Claude, Gemini, Grok, etc.)

Return ONLY the JSON. No explanation, no markdown, no code blocks."""


def _build_prompt(model_data: dict) -> str:
    modalities = model_data.get("modalities", {})
    return ENRICHMENT_PROMPT.format(
        name=model_data.get("name", ""),
        model_id=model_data.get("id", ""),
        provider=model_data.get("provider", ""),
        description=model_data.get("description", "")[:400],
        pipeline_tag=model_data.get("pipeline_tag", "N/A"),
        input_modalities=", ".join(modalities.get("input", [])) or "text",
        output_modalities=", ".join(modalities.get("output", [])) or "text",
        context_window=model_data.get("context_window", 0) or "unknown",
        parameter_count=model_data.get("parameter_count", "unknown"),
        source=model_data.get("source", ""),
    )


def _parse_response(raw: str) -> dict | None:
    """Parse JSON from LLM response. Handles minor formatting issues."""
    raw = raw.strip()
    # Strip markdown code blocks if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON object if there's surrounding text
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    return None


def _merge_enrichment(model: dict, enriched: dict) -> dict:
    """Merge enricher output into model — only fills fields that are None/empty."""
    if enriched.get("capability_profile"):
        model["capability_profile"] = enriched["capability_profile"]

    # Only fill type/category/open_source if not already set (None means not set)
    if model.get("type") is None and enriched.get("type"):
        model["type"] = enriched["type"]

    if model.get("category") is None and enriched.get("category"):
        model["category"] = enriched["category"]

    if model.get("open_source") is None and enriched.get("open_source") is not None:
        model["open_source"] = enriched["open_source"]

    # Merge tags — add enricher tags that aren't already present
    existing_tags = set(model.get("tags", []))
    new_tags = [t for t in enriched.get("tags", []) if t not in existing_tags]
    model["tags"] = list(existing_tags) + new_tags

    return model


def _is_daily_limit(err: str) -> bool:
    """Return True if this 429 is a daily/quota exhaustion (not a per-minute rate limit).

    Daily limit = no point retrying for hours. Stop immediately and resume tomorrow.
    RPM limit   = wait 60s and retry — will resolve within a minute.
    """
    e = err.lower()
    return (
        "per day" in e or "tokens per day" in e or "tpd" in e
        or "per_day" in e or "daily" in e
        or "requests per day" in e or "rpd" in e
    )


def _generate_groq(client, model_data: dict, retries: int = 4) -> dict | None:
    """Call Groq API and return parsed JSON enrichment.

    Raises _DailyLimitReached if the daily quota is exhausted — caller must stop
    the loop, save pending progress, and exit cleanly.
    """
    prompt = _build_prompt(model_data)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=350,
            )
            raw = response.choices[0].message.content.strip()
            parsed = _parse_response(raw)
            if parsed:
                return parsed
            print(f"  [WARN] Could not parse JSON for '{model_data.get('id')}': {raw[:100]}")
            return None
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower() or "rate limit" in err.lower():
                if _is_daily_limit(err):
                    # Daily quota gone — raising stops the whole enrichment loop immediately
                    raise _DailyLimitReached(err[:300])
                # Per-minute rate limit — wait and retry
                wait = 60 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s before retry {attempt + 1}/{retries}...")
                time.sleep(wait)
            else:
                print(f"  [WARN] Groq failed for '{model_data.get('id')}': {e}")
                return None
    print(f"  [WARN] Gave up on '{model_data.get('id')}' after {retries} retries")
    return None


def _generate_ollama(model_data: dict) -> dict | None:
    """Call local Ollama and return parsed JSON enrichment."""
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage

        ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
        llm = ChatOllama(model=ollama_model, temperature=0.1)
        prompt = _build_prompt(model_data)
        result = llm.invoke([HumanMessage(content=prompt)])
        raw = result.content.strip() if hasattr(result, "content") else str(result).strip()
        # Strip qwen3 thinking tags
        if "<think>" in raw and "</think>" in raw:
            raw = raw[raw.index("</think>") + len("</think>"):].strip()
        return _parse_response(raw)
    except Exception as e:
        print(f"  [WARN] Ollama failed for '{model_data.get('id')}': {e}")
        return None


def enrich_models(models: list[dict], save_callback=None, save_every: int = 10) -> list[dict]:
    """Enrich models with capability profiles + type/category/tags/open_source via LLM.

    Skips models that already have a capability_profile.

    Args:
        models: list of model dicts to enrich (in-place)
        save_callback: optional fn(list[dict]) called every save_every enriched models
                       — pass upsert_models here to save progress incrementally so
                       crashes / rate-limit kills never lose work
        save_every: how many enriched models to accumulate before calling save_callback
    """
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    use_groq = provider == "groq"

    if use_groq:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        from groq import Groq
        groq_client = Groq(api_key=api_key)
        print(f"  Using Groq ({GROQ_MODEL}) for enrichment")
    else:
        ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
        print(f"  Using Ollama ({ollama_model}) for enrichment")

    to_enrich = [m for m in models if not m.get("capability_profile")]
    already_done = len(models) - len(to_enrich)

    if already_done:
        print(f"  Skipping {already_done} already-enriched models")
    if not to_enrich:
        print("  All models already enriched.")
        return models

    if use_groq:
        est = len(to_enrich) * DELAY_BETWEEN_REQUESTS / 60
        print(f"  Enriching {len(to_enrich)} models via Groq @ {REQUESTS_PER_MINUTE}/min (est. {est:.0f} min)...")
        if save_callback:
            print(f"  Progress saved every {save_every} models — safe to interrupt and resume.")
        else:
            print(f"  Auto-retries on rate limit. Safe to leave running.")
    else:
        print(f"  Enriching {len(to_enrich)} models via Ollama...")

    enriched_count = 0
    failed_count = 0
    pending_save: list[dict] = []

    for i, model in enumerate(to_enrich):
        try:
            result = _generate_groq(groq_client, model) if use_groq else _generate_ollama(model)
        except _DailyLimitReached as e:
            # Daily quota exhausted — save whatever we have and stop immediately.
            # No point continuing: every remaining model would also fail, wasting hours.
            if save_callback and pending_save:
                save_callback(pending_save)
            print(f"\n  [DAILY LIMIT] Groq daily quota reached after {enriched_count} models.")
            print(f"  {len(pending_save) if pending_save else 0} pending models saved to DB.")
            print(f"\n  ALL progress is saved. Resume tomorrow with:")
            print(f"  python -m src.updater.enrich_only")
            print(f"  (It will skip the {enriched_count} already-enriched models automatically.)\n")
            return models

        if result:
            _merge_enrichment(model, result)
            enriched_count += 1
            if save_callback:
                pending_save.append(model)
        else:
            failed_count += 1

        # Save progress to DB every save_every enriched models
        if save_callback and len(pending_save) >= save_every:
            save_callback(pending_save)
            print(f"  [SAVED] {enriched_count} enriched so far (batch of {len(pending_save)} saved to DB)")
            pending_save = []

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(to_enrich)} ({enriched_count} enriched, {failed_count} failed)")

        if use_groq and i < len(to_enrich) - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    # Save any remaining enriched models
    if save_callback and pending_save:
        save_callback(pending_save)
        print(f"  [SAVED] Final batch of {len(pending_save)} models saved to DB")

    print(f"  Enrichment complete: {enriched_count} enriched, {failed_count} failed")
    return models
