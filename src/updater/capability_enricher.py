"""LLM-generated capability profiles for AI models.

Supports four enrichment backends:
- Cerebras (recommended) — set LLM_PROVIDER=cerebras + CEREBRAS_API_KEY in .env
  Free tier: 60K TPM → all 1700 models done in ~45 minutes
- Gemini — set LLM_PROVIDER=gemini + GEMINI_API_KEY in .env
  Free tier: 15 RPM, 1M tokens/day (may be region-restricted)
- Groq — set LLM_PROVIDER=groq + GROQ_API_KEY in .env
  Free tier: ~100 models/day (hits daily limit fast)
- Ollama (local fallback) — set LLM_PROVIDER=ollama

The enricher does two things in one LLM call:
1. Generates a capability profile (for semantic search)
2. Assigns type, category, tags, open_source (since syncs no longer hardcode these)
"""

import os
import time
import json


class _DailyLimitReached(Exception):
    """Daily token quota exhausted — no point retrying until tomorrow."""
    pass


GROQ_MODEL = "llama-3.3-70b-versatile"
# Groq free tier: 30 RPM, 6000 TPM.
# Each call uses ~500 tokens (prompt ~300 + output ~200).
# 6000 / 500 = 12 calls/min safe → 1 call every 5 seconds.
GROQ_REQUESTS_PER_MINUTE = 12
GROQ_DELAY = 60.0 / GROQ_REQUESTS_PER_MINUTE  # 5 seconds

GEMINI_MODEL = "gemini-2.5-flash"
# Gemini Flash free tier (Google AI Studio): 15 RPM, 1M tokens/day, 1500 RPD.
# At ~500 tokens/call: all 1700 models done in ~2 hours.
GEMINI_REQUESTS_PER_MINUTE = 14  # stay just under 15 RPM limit
GEMINI_DELAY = 60.0 / GEMINI_REQUESTS_PER_MINUTE  # ~4.3 seconds

CEREBRAS_MODEL = "qwen-3-235b-a22b-instruct-2507"
# Cerebras free tier: 60K TPM, 30 RPM.
# At ~500 tokens/call: safe at 30 calls/min → all 1700 models in ~45 minutes.
CEREBRAS_REQUESTS_PER_MINUTE = 5  # very conservative — no rate limit retries, steady flow
CEREBRAS_DELAY = 60.0 / CEREBRAS_REQUESTS_PER_MINUTE  # 12 seconds

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
  "type": "the primary function of this model — what it fundamentally does (e.g., text generation, embedding, image generation, audio processing, reranking). Use the most accurate description, not a forced category.",
  "category": "2-3 short tags describing its primary strengths for search and filtering (e.g., code generation, long-context, reasoning, multilingual, medical, legal). Choose tags that would help someone find this model when they need it.",
  "tags": ["list", "of", "relevant", "capability", "tags"],
  "open_source": true or false
}}

Rules:
- capability_profile: be task-specific. Never say "general-purpose". Mention actual domains.
- type: describe what the model fundamentally does. A chat model generates text in conversation. An embedding model produces vectors. A multimodal model processes multiple input/output types together. Reason from the model's actual function, not a fixed list.
- category: choose tags that reflect genuine strengths — what would someone be searching for when this model is the right answer? Think about the user's need, not the model's marketing.
- tags: include any that genuinely apply: vision, audio, code, instruct, multilingual, long-context, function-calling, reasoning, math, medical, legal, embedding, image-generation, speech-recognition, text-to-speech, file-input, and any others that accurately describe this model's capabilities.
- open_source: true if the model weights are publicly available for anyone to download and run (regardless of license). False if the weights are proprietary and only accessible via API.

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
    # Groq daily signals
    if "per day" in e or "tokens per day" in e or "tpd" in e or "per_day" in e or "daily" in e:
        return True
    if "requests per day" in e or "rpd" in e:
        return True
    # Gemini-specific daily signals (must say "perday" or quota ID with day)
    if "perday" in e or "per_day" in e or "requestsperday" in e:
        return True
    # Do NOT treat generic "quota" or "resource_exhausted" as daily — those are RPM errors
    return False


def _generate_cerebras(client, model_data: dict, retries: int = 4) -> dict | None:
    """Call Cerebras API and return parsed JSON enrichment.

    Uses Cerebras free tier: 60K TPM, 30 RPM.
    Raises _DailyLimitReached on quota exhaustion.
    """
    prompt = _build_prompt(model_data)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=CEREBRAS_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=350,
            )
            raw = response.choices[0].message.content.strip()
            # Strip qwen thinking tags if present
            if "<think>" in raw and "</think>" in raw:
                raw = raw[raw.index("</think>") + len("</think>"):].strip()
            parsed = _parse_response(raw)
            if parsed:
                return parsed
            print(f"  [WARN] Could not parse JSON for '{model_data.get('id')}': {raw[:100]}")
            return None
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower() or "rate limit" in err.lower():
                if _is_daily_limit(err):
                    raise _DailyLimitReached(err[:300])
                wait = 60 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s before retry {attempt + 1}/{retries}...")
                time.sleep(wait)
            else:
                print(f"  [WARN] Cerebras failed for '{model_data.get('id')}': {e}")
                return None
    print(f"  [WARN] Gave up on '{model_data.get('id')}' after {retries} retries")
    return None


def _generate_gemini(client, model_data: dict, retries: int = 4) -> dict | None:
    """Call Gemini Flash via OpenAI-compatible API and return parsed JSON enrichment.

    Uses Google AI Studio free tier: 15 RPM, 1M tokens/day, 1500 requests/day.
    Raises _DailyLimitReached on quota exhaustion.
    """
    prompt = _build_prompt(model_data)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,  # Gemini 2.5 Flash is a thinking model — needs extra tokens for reasoning
            )
            raw = response.choices[0].message.content.strip()
            parsed = _parse_response(raw)
            if parsed:
                return parsed
            print(f"  [WARN] Could not parse JSON for '{model_data.get('id')}': {raw[:100]}")
            return None
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower() or "rate limit" in err.lower() or "quota" in err.lower():
                if _is_daily_limit(err):
                    raise _DailyLimitReached(err[:300])
                wait = 60 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s before retry {attempt + 1}/{retries}...")
                time.sleep(wait)
            else:
                print(f"  [WARN] Gemini failed for '{model_data.get('id')}': {e}")
                return None
    print(f"  [WARN] Gave up on '{model_data.get('id')}' after {retries} retries")
    return None


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
    use_cerebras = provider == "cerebras"
    use_groq = provider == "groq"
    use_gemini = provider == "gemini"

    llm_client = None

    if use_cerebras:
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise RuntimeError("CEREBRAS_API_KEY not set. Get a free key at https://cloud.cerebras.ai")
        from openai import OpenAI
        llm_client = OpenAI(
            api_key=api_key,
            base_url="https://api.cerebras.ai/v1",
        )
        print(f"  Using Cerebras ({CEREBRAS_MODEL}) for enrichment")
    elif use_gemini:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey")
        from openai import OpenAI
        llm_client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        print(f"  Using Gemini Flash ({GEMINI_MODEL}) for enrichment")
    elif use_groq:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        from groq import Groq
        llm_client = Groq(api_key=api_key)
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

    if use_cerebras:
        est = len(to_enrich) * CEREBRAS_DELAY / 60
        print(f"  Enriching {len(to_enrich)} models via Cerebras @ {CEREBRAS_REQUESTS_PER_MINUTE}/min (est. {est:.0f} min)...")
    elif use_gemini:
        est = len(to_enrich) * GEMINI_DELAY / 60
        print(f"  Enriching {len(to_enrich)} models via Gemini @ {GEMINI_REQUESTS_PER_MINUTE}/min (est. {est:.0f} min)...")
    elif use_groq:
        est = len(to_enrich) * GROQ_DELAY / 60
        print(f"  Enriching {len(to_enrich)} models via Groq @ {GROQ_REQUESTS_PER_MINUTE}/min (est. {est:.0f} min)...")
    else:
        print(f"  Enriching {len(to_enrich)} models via Ollama...")

    if save_callback:
        print(f"  Progress saved every {save_every} models — safe to interrupt and resume.")

    enriched_count = 0
    failed_count = 0
    pending_save: list[dict] = []

    for i, model in enumerate(to_enrich):
        try:
            if use_cerebras:
                result = _generate_cerebras(llm_client, model)
            elif use_gemini:
                result = _generate_gemini(llm_client, model)
            elif use_groq:
                result = _generate_groq(llm_client, model)
            else:
                result = _generate_ollama(model)
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

        if i < len(to_enrich) - 1:
            if use_cerebras:
                time.sleep(CEREBRAS_DELAY)
            elif use_gemini:
                time.sleep(GEMINI_DELAY)
            elif use_groq:
                time.sleep(GROQ_DELAY)

    # Save any remaining enriched models
    if save_callback and pending_save:
        save_callback(pending_save)
        print(f"  [SAVED] Final batch of {len(pending_save)} models saved to DB")

    print(f"  Enrichment complete: {enriched_count} enriched, {failed_count} failed")
    return models
