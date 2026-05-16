"""Tools for the Analysis Agent: model search, comparison, and details."""

import json
from langchain_core.tools import tool
from src.db import qdrant_manager


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


@tool
def search_models(query: str, filters: str | dict = "{}") -> str:
    """Search for AI models using semantic search with optional metadata filters.

    Args:
        query: Natural language description of what you need (e.g., "fast chat model for code generation")
        filters: JSON string of filters. Supported keys:
            - has_vision: true — only models that accept image input
            - has_function_calling: true — only models with tools/function-calling support
            - has_reasoning: true — only models with extended reasoning/thinking mode
            - has_audio: true — only models that accept or produce audio
            - has_image_generation: true — only models that produce images
            - open_source: true/false
            - min_context_window: minimum context window size (integer)
            - max_input_price: max price per 1M input tokens (float)
            - provider: model provider/org name (e.g., "openai", "anthropic", "meta-llama")
            - platform: API platform where model is available (e.g., "openai", "groq", "together_ai", "openrouter")

    Returns:
        Top matching models with scores and key details
    """
    filter_dict = _parse_filters(filters)
    results = qdrant_manager.semantic_search(query, filter_dict, top_k=10)

    if not results:
        return "No models found matching your query and filters. Try broadening your search."

    output_lines = [f"Found {len(results)} matching models:\n"]
    for i, model in enumerate(results, 1):
        score = model.get("_score", 0)
        name = model.get("name", "Unknown")
        model_id = model.get("id", "")
        ctx = model.get("context_window", 0)
        input_price = model.get("input_price_per_mtok", 0)
        output_price = model.get("output_price_per_mtok", 0)
        open_src = model.get("open_source")
        desc = model.get("description", "")[:150]

        # Build capability flags string
        caps = []
        if model.get("has_vision"):
            caps.append("vision")
        if model.get("has_function_calling"):
            caps.append("function-calling")
        if model.get("has_reasoning"):
            caps.append("reasoning")
        if model.get("has_audio"):
            caps.append("audio")
        if model.get("has_image_generation"):
            caps.append("image-gen")
        caps_str = ", ".join(caps) if caps else "text-only"

        open_src_str = "Yes" if open_src is True else ("No" if open_src is False else "Unknown")

        output_lines.append(f"{i}. **{name}** (ID: {model_id})")
        output_lines.append(f"   Score: {score:.3f} | Context: {ctx:,} tokens")
        if input_price > 0:
            output_lines.append(f"   Pricing: ${input_price:.2f}/${output_price:.2f} per 1M tokens (in/out)")
        else:
            output_lines.append(f"   Pricing: Free / self-hosted")
        output_lines.append(f"   Capabilities: {caps_str} | Open Source: {open_src_str}")
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
    open_src = model.get("open_source")
    open_src_str = "Yes" if open_src is True else ("No" if open_src is False else "Unknown")
    lines.append(f"- **Open Source**: {open_src_str}")

    # Capability flags
    caps = []
    if model.get("has_vision"):
        caps.append("vision (image input)")
    if model.get("has_function_calling"):
        caps.append("function calling")
    if model.get("has_reasoning"):
        caps.append("extended reasoning")
    if model.get("has_audio"):
        caps.append("audio")
    if model.get("has_image_generation"):
        caps.append("image generation")
    lines.append(f"- **Capabilities**: {', '.join(caps) if caps else 'text generation'}")
    lines.append(f"- **Context Window**: {model.get('context_window', 0):,} tokens")

    input_price = model.get("input_price_per_mtok", 0)
    output_price = model.get("output_price_per_mtok", 0)
    if input_price > 0:
        lines.append(f"- **Input Price**: ${input_price:.4f} per 1M tokens")
        lines.append(f"- **Output Price**: ${output_price:.4f} per 1M tokens")
    else:
        lines.append("- **Pricing**: Free / self-hosted (no API pricing)")

    # Per-platform pricing array — critical for cost agent to pick cheapest platform
    pricing_array = model.get("pricing", [])
    if pricing_array:
        lines.append(f"\n**Per-Platform Pricing** (input/output per 1M tokens):")
        for entry in pricing_array:
            plat = entry.get("platform", "?")
            ip = entry.get("input_price_per_mtok", 0)
            op = entry.get("output_price_per_mtok", 0)
            if ip > 0 or op > 0:
                lines.append(f"  - {plat}: ${ip:.4f} in / ${op:.4f} out")

    available_platforms = model.get("available_platforms", [])
    if available_platforms:
        lines.append(f"- **Available Platforms**: {', '.join(available_platforms)}")

    modalities = model.get("modalities", {})
    if modalities:
        lines.append(f"- **Input Modalities**: {', '.join(modalities.get('input', []))}")
        lines.append(f"- **Output Modalities**: {', '.join(modalities.get('output', []))}")

    if model.get("param_count"):
        lines.append(f"- **Parameters**: {model['param_count']}B")

    if model.get("knowledge_cutoff"):
        lines.append(f"- **Knowledge Cutoff**: {model['knowledge_cutoff']}")

    desc = model.get("description", "")
    if desc:
        lines.append(f"\n**Description**: {desc[:500]}")

    return "\n".join(lines)


@tool
def compare_models(model_ids: list[str]) -> str:
    """Compare multiple models side by side.

    Args:
        model_ids: List of model IDs to compare (e.g., ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"])

    Returns:
        Side-by-side comparison table
    """
    models = qdrant_manager.get_models(model_ids)

    if not models:
        return "None of the specified models were found. Try searching first with search_models."

    if len(models) == 1:
        return f"Only found 1 of {len(model_ids)} models. Found: {models[0].get('name')}. Others not in database."

    # Build comparison
    lines = ["## Model Comparison\n"]
    lines.append("| Feature | " + " | ".join(m.get("name", "?") for m in models) + " |")
    lines.append("|" + "---|" * (len(models) + 1))

    def _caps(m: dict) -> str:
        flags = []
        if m.get("has_vision"):
            flags.append("vision")
        if m.get("has_function_calling"):
            flags.append("fn-call")
        if m.get("has_reasoning"):
            flags.append("reasoning")
        if m.get("has_audio"):
            flags.append("audio")
        if m.get("has_image_generation"):
            flags.append("img-gen")
        return ", ".join(flags) if flags else "text"

    rows = [
        ("Provider", lambda m: m.get("provider", "?")),
        ("Context Window", lambda m: f"{m.get('context_window', 0):,}"),
        ("Input $/1M tok", lambda m: f"${m.get('input_price_per_mtok', 0):.2f}"),
        ("Output $/1M tok", lambda m: f"${m.get('output_price_per_mtok', 0):.2f}"),
        ("Open Source", lambda m: "Yes" if m.get("open_source") is True else ("No" if m.get("open_source") is False else "Unknown")),
        ("Capabilities", _caps),
    ]

    for label, getter in rows:
        lines.append(f"| {label} | " + " | ".join(getter(m) for m in models) + " |")

    return "\n".join(lines)


@tool
def save_recommendations(model_ids: list[str], reasoning: str = "") -> str:
    """Save the final list of recommended model IDs for the Cost Agent.

    Call this once at the end after you have finished evaluating models.
    This is the ONLY way the Cost Agent receives your recommendations —
    do not write model IDs as free text instead of calling this tool.

    Args:
        model_ids: List of exact model IDs to recommend (e.g., ["openai/gpt-4o", "anthropic/claude-3-5-sonnet"])
        reasoning: Brief summary of why these models were chosen

    Returns:
        Confirmation that recommendations were saved
    """
    if not model_ids:
        return '{"status": "error", "message": "No model IDs provided. Pass at least one model ID."}'
    return json.dumps({
        "status": "saved",
        "recommended_models": model_ids,
        "count": len(model_ids),
        "reasoning": reasoning,
        "message": f"{len(model_ids)} model(s) saved for cost estimation.",
    })


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
