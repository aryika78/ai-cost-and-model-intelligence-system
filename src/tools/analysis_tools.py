"""Tools for the Analysis Agent: model search, comparison, and details."""

import os
import json
from groq import Groq
from langchain_core.tools import tool
from src.db import qdrant_manager

# Lazy Groq client for query expansion / re-ranking
_groq_client = None


def _get_groq():
    global _groq_client
    if os.environ.get("LLM_PROVIDER", "ollama").lower() == "ollama":
        return None
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _expand_query(query: str) -> list[str]:
    """Use Groq to generate 3 alternative search queries for better recall."""
    client = _get_groq()
    if not client:
        return []
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": (
                    f"I'm searching for AI models matching: \"{query}\"\n\n"
                    "Generate exactly 3 alternative search queries that would help find "
                    "the right models. Focus on different angles: capabilities, technical "
                    "features, and use-case descriptions.\n\n"
                    "Respond with ONLY 3 lines, one query per line, no numbering or bullets."
                ),
            }],
            temperature=0.4,
            max_tokens=150,
        )
        lines = [l.strip() for l in response.choices[0].message.content.strip().split("\n") if l.strip()]
        return lines[:3]
    except Exception:
        return []


def _parse_filters(filters) -> dict:
    """Accept filters as either a JSON string or a dict from tool-calling models."""
    if not filters or filters == "{}":
        return {}
    if isinstance(filters, dict):
        return filters
    try:
        return json.loads(filters)
    except (json.JSONDecodeError, TypeError):
        return {}


def _multi_query_search(query: str, filter_dict: dict, top_k: int = 15) -> list[dict]:
    """Search with original query + expanded queries, merge by max score."""
    expanded = _expand_query(query)
    all_queries = [query] + expanded

    # Collect results from all queries, keyed by model ID
    seen: dict[str, dict] = {}
    for q in all_queries:
        results = qdrant_manager.semantic_search(q, filter_dict, top_k=top_k)
        for r in results:
            mid = r.get("id", "")
            if mid not in seen or r.get("_score", 0) > seen[mid].get("_score", 0):
                seen[mid] = r

    # Sort by best score descending
    merged = sorted(seen.values(), key=lambda x: x.get("_score", 0), reverse=True)
    return merged[:top_k]


def _llm_rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """Ask Groq to re-rank candidates by actual task fit."""
    client = _get_groq()
    if not client or not candidates:
        return candidates[:top_n]

    # Build compact candidate list for the LLM
    candidate_lines = []
    for i, m in enumerate(candidates):
        name = m.get("name", "Unknown")
        mid = m.get("id", "")
        cap = m.get("capability_profile", m.get("description", ""))[:200]
        candidate_lines.append(f"{i}: {name} (ID: {mid}) — {cap}")

    candidates_text = "\n".join(candidate_lines)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": (
                    f"User task: \"{query}\"\n\n"
                    f"Candidate AI models:\n{candidates_text}\n\n"
                    f"Rank the top {top_n} models that best fit the user's task. "
                    "Consider actual capability match, not just name similarity.\n\n"
                    f"Respond with ONLY {top_n} numbers (the indices), one per line, "
                    "best fit first. No explanation."
                ),
            }],
            temperature=0.1,
            max_tokens=50,
        )
        lines = response.choices[0].message.content.strip().split("\n")
        indices = []
        for line in lines:
            # Extract the number from each line
            num = "".join(c for c in line.split()[0] if c.isdigit()) if line.strip() else ""
            if num and int(num) < len(candidates):
                indices.append(int(num))
        if len(indices) >= 2:
            # Deduplicate while preserving order
            seen_idx = set()
            unique_indices = []
            for idx in indices:
                if idx not in seen_idx:
                    seen_idx.add(idx)
                    unique_indices.append(idx)
            return [candidates[i] for i in unique_indices[:top_n]]
    except Exception:
        pass

    return candidates[:top_n]


@tool
def search_models(query: str, filters: str | dict = "{}") -> str:
    """Search for AI models using semantic search with optional metadata filters.

    Args:
        query: Natural language description of what you need (e.g., "fast chat model for code generation")
        filters: JSON string of filters. Supported keys:
            - type: "chat", "embedding", "image_generation", "audio"
            - category: "general", "code", "long_context", "embedding", etc.
            - tags: list of tags to match (e.g., ["vision", "function_calling"])
            - open_source: true/false
            - min_context_window: minimum context window size
            - max_input_price: max price per 1M input tokens
            - provider: model provider name

    Returns:
        Top matching models with scores and key details
    """
    filter_dict = _parse_filters(filters)

    # Layer 2: Multi-query search (query expansion) + LLM re-ranking
    # Falls back to plain search if Groq unavailable
    if _get_groq():
        candidates = _multi_query_search(query, filter_dict, top_k=15)
        results = _llm_rerank(query, candidates, top_n=10)
    else:
        results = qdrant_manager.semantic_search(query, filter_dict, top_k=10)

    if not results:
        return "No models found matching your query and filters. Try broadening your search."

    output_lines = [f"Found {len(results)} matching models:\n"]
    for i, model in enumerate(results, 1):
        score = model.pop("_score", 0)
        name = model.get("name", "Unknown")
        model_id = model.get("id", "")
        ctx = model.get("context_window", 0)
        input_price = model.get("input_price_per_mtok", 0)
        output_price = model.get("output_price_per_mtok", 0)
        tags = model.get("tags", [])
        open_src = model.get("open_source", False)
        desc = model.get("description", "")[:150]

        output_lines.append(f"{i}. **{name}** (ID: {model_id})")
        output_lines.append(f"   Score: {score:.3f} | Context: {ctx:,} tokens")
        if input_price > 0:
            output_lines.append(f"   Pricing: ${input_price:.2f}/${output_price:.2f} per 1M tokens (in/out)")
        else:
            output_lines.append(f"   Pricing: Free / self-hosted")
        output_lines.append(f"   Tags: {', '.join(tags)} | Open Source: {open_src}")
        if desc:
            output_lines.append(f"   Description: {desc}")
        output_lines.append("")

    return "\n".join(output_lines)


@tool
def get_model_details(model_id: str) -> str:
    """Get full details for a specific model by its ID.

    Args:
        model_id: The model identifier (e.g., "openai/gpt-4o", "meta-llama/llama-3.1-70b-instruct")

    Returns:
        Complete model information including pricing, capabilities, and metadata
    """
    model = qdrant_manager.get_model(model_id)

    if not model:
        return f"Model '{model_id}' not found in the database. Try searching with search_models first."

    lines = [f"## Model Details: {model.get('name', model_id)}\n"]
    lines.append(f"- **ID**: {model.get('id', '')}")
    lines.append(f"- **Provider**: {model.get('provider', 'Unknown')}")
    lines.append(f"- **Type**: {model.get('type', 'Unknown')}")
    lines.append(f"- **Category**: {model.get('category', 'Unknown')}")
    lines.append(f"- **Open Source**: {model.get('open_source', False)}")
    lines.append(f"- **Context Window**: {model.get('context_window', 0):,} tokens")

    input_price = model.get("input_price_per_mtok", 0)
    output_price = model.get("output_price_per_mtok", 0)
    if input_price > 0:
        lines.append(f"- **Input Price**: ${input_price:.4f} per 1M tokens")
        lines.append(f"- **Output Price**: ${output_price:.4f} per 1M tokens")
    else:
        lines.append("- **Pricing**: Free / self-hosted (no API pricing)")

    tags = model.get("tags", [])
    if tags:
        lines.append(f"- **Tags**: {', '.join(tags)}")

    modalities = model.get("modalities", {})
    if modalities:
        lines.append(f"- **Input Modalities**: {', '.join(modalities.get('input', []))}")
        lines.append(f"- **Output Modalities**: {', '.join(modalities.get('output', []))}")

    desc = model.get("description", "")
    if desc:
        lines.append(f"\n**Description**: {desc[:500]}")

    if model.get("parameter_count"):
        lines.append(f"- **Parameters**: {model['parameter_count']}")

    return "\n".join(lines)


@tool
def compare_models(model_ids: str) -> str:
    """Compare multiple models side by side.

    Args:
        model_ids: Comma-separated list of model IDs to compare (e.g., "openai/gpt-4o,anthropic/claude-3.5-sonnet")

    Returns:
        Side-by-side comparison table
    """
    ids = [mid.strip() for mid in model_ids.split(",")]
    models = qdrant_manager.get_models(ids)

    if not models:
        return "None of the specified models were found. Try searching first with search_models."

    if len(models) == 1:
        return f"Only found 1 of {len(ids)} models. Found: {models[0].get('name')}. Others not in database."

    # Build comparison
    lines = ["## Model Comparison\n"]
    lines.append("| Feature | " + " | ".join(m.get("name", "?") for m in models) + " |")
    lines.append("|" + "---|" * (len(models) + 1))

    rows = [
        ("Provider", lambda m: m.get("provider", "?")),
        ("Type", lambda m: m.get("type", "?")),
        ("Context Window", lambda m: f"{m.get('context_window', 0):,}"),
        ("Input $/1M tok", lambda m: f"${m.get('input_price_per_mtok', 0):.2f}"),
        ("Output $/1M tok", lambda m: f"${m.get('output_price_per_mtok', 0):.2f}"),
        ("Open Source", lambda m: str(m.get("open_source", False))),
        ("Tags", lambda m: ", ".join(m.get("tags", [])[:5])),
    ]

    for label, getter in rows:
        lines.append(f"| {label} | " + " | ".join(getter(m) for m in models) + " |")

    return "\n".join(lines)


@tool
def count_matching_models(query: str, filters: str | dict = "{}") -> str:
    """Count how many models match the given query and filters. Useful to check if filters are too strict.

    Args:
        query: Search query
        filters: JSON string of filters (same format as search_models)

    Returns:
        Count of matching models
    """
    filter_dict = _parse_filters(filters)

    count = qdrant_manager.count_results(query, filter_dict)
    return f"Found {count} models matching query='{query}' with filters={json.dumps(filter_dict)}."
