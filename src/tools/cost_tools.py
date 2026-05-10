"""Tools for the Cost Agent: API cost, self-hosting, GPU options, embeddings, fine-tuning, scenarios."""

import json
import os
from langchain_core.tools import tool
from src.db import qdrant_manager

_gpu_data = None
_platforms_data = None


def _load_gpu_data() -> dict:
    global _gpu_data
    if _gpu_data is None:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "gpu_pricing.json")
        with open(os.path.normpath(path)) as f:
            _gpu_data = json.load(f)
    return _gpu_data


def _load_platforms_data() -> dict:
    global _platforms_data
    if _platforms_data is None:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "platforms.json")
        with open(os.path.normpath(path)) as f:
            _platforms_data = json.load(f)
    return _platforms_data


@tool
def calculate_api_cost(params: str) -> str:
    """Calculate the cost of using an AI model via API.

    Args:
        params: JSON string with:
            - model_id: str (model identifier to look up pricing)
            - avg_input_tokens: int (average input tokens per request)
            - avg_output_tokens: int (average output tokens per request)
            - requests_per_day: int
            - cache_hit_rate: float (0-1, fraction of requests using cached prompt)
            - batch_percentage: float (0-1, fraction using batch API)
            - conversation_turns: int (average turns per conversation, for context accumulation)
            - agent_calls_per_request: int (if using agents, how many LLM calls per user request)

    Returns:
        Detailed cost breakdown with daily, monthly, yearly estimates
    """
    try:
        p = json.loads(params)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON params - {e}"

    model_id = p.get("model_id", "")
    avg_input = p.get("avg_input_tokens", 1000)
    avg_output = p.get("avg_output_tokens", 500)
    rpd = p.get("requests_per_day", 1000)
    cache_rate = p.get("cache_hit_rate", 0.0)
    batch_pct = p.get("batch_percentage", 0.0)
    turns = p.get("conversation_turns", 1)
    agent_calls = p.get("agent_calls_per_request", 1)

    # Look up model pricing
    model = qdrant_manager.get_model(model_id)
    if model:
        input_price = model.get("input_price_per_mtok", 0)
        output_price = model.get("output_price_per_mtok", 0)
        model_name = model.get("name", model_id)
    else:
        return (
            f"Model '{model_id}' not found in database. Cannot calculate cost without pricing data. "
            f"Please use search_models to find the correct model ID first."
        )

    if input_price == 0 and output_price == 0:
        return (
            f"Model '{model_name}' has no API pricing (it's likely open-source/self-hosted). "
            f"Use calculate_self_hosting_cost instead."
        )

    # Calculate effective tokens per request accounting for conversation turns
    # In multi-turn: input grows as context accumulates
    if turns > 1:
        # Average input across turns: turn 1 = base, turn N = base + (N-1) * (avg_input + avg_output)
        avg_accumulated_input = avg_input + (turns - 1) * (avg_input + avg_output) / 2
        effective_input = avg_accumulated_input
    else:
        effective_input = avg_input

    # Agent multiplier
    effective_input *= agent_calls
    effective_output = avg_output * agent_calls
    effective_rpd = rpd

    # Calculate base cost per request
    input_cost_per_req = (effective_input / 1_000_000) * input_price
    output_cost_per_req = (effective_output / 1_000_000) * output_price
    base_cost_per_req = input_cost_per_req + output_cost_per_req

    # Apply discounts
    # Cached requests: 90% discount on input (Anthropic-style) or 75% (Google-style), use 50% as conservative
    cached_requests = rpd * cache_rate
    non_cached_requests = rpd * (1 - cache_rate)
    cache_discount_factor = 0.5  # 50% discount on cached input tokens

    # Batch requests: 50% discount
    batch_requests = rpd * batch_pct
    non_batch = rpd * (1 - batch_pct)
    batch_discount_factor = 0.5

    # Effective daily cost
    daily_cost = 0

    # Non-cached, non-batch requests (full price)
    regular_fraction = max(0, 1 - cache_rate - batch_pct)
    daily_cost += rpd * regular_fraction * base_cost_per_req

    # Cached requests
    cached_input_cost = (effective_input / 1_000_000) * input_price * cache_discount_factor
    cached_cost_per_req = cached_input_cost + output_cost_per_req
    daily_cost += cached_requests * cached_cost_per_req

    # Batch requests
    batch_cost_per_req = base_cost_per_req * batch_discount_factor
    daily_cost += batch_requests * batch_cost_per_req

    monthly_cost = daily_cost * 30
    yearly_cost = daily_cost * 365

    lines = [
        f"## API Cost Estimate: {model_name}\n",
        f"### Assumptions",
        f"- Input tokens/request: {avg_input:,} (effective with turns: {effective_input:,.0f})",
        f"- Output tokens/request: {avg_output:,} (effective: {effective_output:,.0f})",
        f"- Requests/day: {rpd:,}",
        f"- Conversation turns: {turns}",
        f"- Agent calls/request: {agent_calls}",
        f"- Cache hit rate: {cache_rate:.0%}",
        f"- Batch API usage: {batch_pct:.0%}",
        f"",
        f"### Pricing",
        f"- Input: ${input_price:.4f} per 1M tokens",
        f"- Output: ${output_price:.4f} per 1M tokens",
        f"",
        f"### Cost Breakdown",
        f"- Cost per request (base): ${base_cost_per_req:.6f}",
        f"- **Daily cost: ${daily_cost:.2f}**",
        f"- **Monthly cost (30d): ${monthly_cost:.2f}**",
        f"- **Yearly cost (365d): ${yearly_cost:.2f}**",
    ]

    return "\n".join(lines)


@tool
def calculate_self_hosting_cost(params: str) -> str:
    """Calculate the cost of self-hosting an AI model on rented GPUs.

    Args:
        params: JSON string with:
            - model_size: str (e.g., "7b", "13b", "70b", "405b")
            - quantization: str ("fp16", "int8", "int4")
            - gpu_name: str (e.g., "NVIDIA A100 80GB", "NVIDIA H100 80GB")
            - provider: str (e.g., "runpod", "lambda_labs", "aws")
            - hours_per_day: int (24 for always-on, less for on-demand)
            - redundancy: int (number of replicas, minimum 1)

    Returns:
        Self-hosting cost breakdown
    """
    try:
        p = json.loads(params)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON params - {e}"

    model_size = p.get("model_size", "7b")
    quant = p.get("quantization", "fp16")
    gpu_name = p.get("gpu_name", "")
    provider = p.get("provider", "runpod")
    hours = p.get("hours_per_day", 24)
    redundancy = p.get("redundancy", 1)

    gpu_data = _load_gpu_data()
    vram_key = f"{model_size}_{quant}"
    vram_needed = gpu_data.get("model_vram_requirements", {}).get(vram_key, 0)

    if not vram_needed:
        return f"Unknown model size/quantization combo: {model_size}/{quant}. Use one of: {list(gpu_data['model_vram_requirements'].keys())}"

    # Find GPU
    gpu_info = None
    for gpu in gpu_data["gpus"]:
        if gpu_name and gpu["name"].lower() == gpu_name.lower():
            gpu_info = gpu
            break
    if not gpu_info and not gpu_name:
        # Auto-select cheapest GPU that fits
        candidates = []
        for gpu in gpu_data["gpus"]:
            if gpu["vram_gb"] >= vram_needed and provider in gpu.get("providers", {}):
                price = gpu["providers"][provider]["hourly_usd"]
                candidates.append((gpu, price))
        if candidates:
            candidates.sort(key=lambda x: x[1])
            gpu_info = candidates[0][0]

    if not gpu_info:
        return f"No suitable GPU found for {vram_needed}GB VRAM requirement with provider '{provider}'."

    # Calculate GPUs needed
    gpus_needed = max(1, -(-vram_needed // gpu_info["vram_gb"]))  # Ceiling division

    provider_info = gpu_info.get("providers", {}).get(provider)
    if not provider_info:
        available = list(gpu_info["providers"].keys())
        return f"GPU '{gpu_info['name']}' not available on '{provider}'. Available: {available}"

    hourly_rate = provider_info["hourly_usd"]
    total_gpus = gpus_needed * redundancy
    hourly_total = hourly_rate * total_gpus
    daily_cost = hourly_total * hours
    monthly_cost = daily_cost * 30
    yearly_cost = daily_cost * 365

    lines = [
        f"## Self-Hosting Cost Estimate\n",
        f"### Configuration",
        f"- Model size: {model_size} ({quant})",
        f"- VRAM needed: {vram_needed} GB",
        f"- GPU: {gpu_info['name']} ({gpu_info['vram_gb']} GB VRAM)",
        f"- GPUs per instance: {gpus_needed}",
        f"- Replicas: {redundancy}",
        f"- Total GPUs: {total_gpus}",
        f"- Provider: {provider}",
        f"- Hours/day: {hours}",
        f"",
        f"### Cost Breakdown",
        f"- Hourly rate per GPU: ${hourly_rate:.2f}",
        f"- Hourly total ({total_gpus} GPUs): ${hourly_total:.2f}",
        f"- **Daily cost: ${daily_cost:.2f}**",
        f"- **Monthly cost (30d): ${monthly_cost:.2f}**",
        f"- **Yearly cost (365d): ${yearly_cost:.2f}**",
    ]

    return "\n".join(lines)


@tool
def get_gpu_options(model_id: str) -> str:
    """Get GPU options that can run a given model, with pricing across providers.

    Args:
        model_id: Model identifier or size descriptor (e.g., "70b", "meta-llama/llama-3.1-70b")

    Returns:
        Table of GPU options with pricing
    """
    gpu_data = _load_gpu_data()

    # Determine VRAM needed
    size_str = model_id.lower()
    vram_needed = None

    # Try to extract size from model ID
    for size_key, vram in gpu_data["model_vram_requirements"].items():
        size_part = size_key.split("_")[0]  # e.g., "70b"
        if size_part in size_str:
            # Default to int4 for inference
            quant = "int4"
            full_key = f"{size_part}_{quant}"
            vram_needed = gpu_data["model_vram_requirements"].get(full_key, vram)
            break

    if not vram_needed:
        vram_needed = 16  # Default for unknown models

    lines = [
        f"## GPU Options for Model: {model_id}",
        f"Estimated VRAM needed: ~{vram_needed} GB (INT4 quantization)\n",
    ]

    for gpu in gpu_data["gpus"]:
        if gpu["vram_gb"] < vram_needed:
            continue

        gpus_needed = max(1, -(-vram_needed // gpu["vram_gb"]))
        lines.append(f"### {gpu['name']} ({gpu['vram_gb']}GB VRAM, {gpus_needed} GPU(s) needed)")

        for provider, info in gpu["providers"].items():
            hourly = info["hourly_usd"] * gpus_needed
            monthly = hourly * 24 * 30
            lines.append(f"  - {provider}: ${hourly:.2f}/hr (${monthly:.0f}/mo always-on)")

        lines.append("")

    return "\n".join(lines)


@tool
def calculate_embedding_cost(params: str) -> str:
    """Calculate embedding cost for RAG pipeline.

    Args:
        params: JSON string with:
            - embedding_model_id: str (e.g., "openai/text-embedding-3-small")
            - num_documents: int
            - avg_tokens_per_doc: int
            - re_embedding_frequency: str ("once", "weekly", "monthly")
            - query_tokens_per_day: int (tokens for query embeddings)

    Returns:
        Embedding cost breakdown
    """
    try:
        p = json.loads(params)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON params - {e}"

    model_id = p.get("embedding_model_id", "openai/text-embedding-3-small")
    num_docs = p.get("num_documents", 1000)
    avg_tokens = p.get("avg_tokens_per_doc", 500)
    re_embed = p.get("re_embedding_frequency", "once")
    query_tokens_day = p.get("query_tokens_per_day", 100000)

    # Look up embedding model pricing
    model = qdrant_manager.get_model(model_id)
    if model and model.get("input_price_per_mtok", 0) > 0:
        price_per_mtok = model["input_price_per_mtok"]
    else:
        # Default embedding prices
        price_per_mtok = 0.02  # $0.02 per 1M tokens (text-embedding-3-small level)

    total_doc_tokens = num_docs * avg_tokens
    initial_embed_cost = (total_doc_tokens / 1_000_000) * price_per_mtok

    re_embed_monthly = 0
    if re_embed == "weekly":
        re_embed_monthly = initial_embed_cost * 4
    elif re_embed == "monthly":
        re_embed_monthly = initial_embed_cost

    query_daily_cost = (query_tokens_day / 1_000_000) * price_per_mtok
    query_monthly_cost = query_daily_cost * 30

    total_monthly = re_embed_monthly + query_monthly_cost

    lines = [
        f"## Embedding Cost Estimate\n",
        f"### Configuration",
        f"- Model: {model_id}",
        f"- Price: ${price_per_mtok:.4f} per 1M tokens",
        f"- Documents: {num_docs:,} ({avg_tokens:,} tokens avg)",
        f"- Total document tokens: {total_doc_tokens:,}",
        f"- Re-embedding: {re_embed}",
        f"- Query volume: {query_tokens_day:,} tokens/day",
        f"",
        f"### Cost Breakdown",
        f"- Initial embedding cost: ${initial_embed_cost:.4f}",
        f"- Re-embedding cost/month: ${re_embed_monthly:.4f}",
        f"- Query embedding cost/month: ${query_monthly_cost:.4f}",
        f"- **Total monthly: ${total_monthly:.4f}**",
        f"- **Total yearly: ${total_monthly * 12:.2f}**",
    ]

    return "\n".join(lines)


@tool
def calculate_finetuning_cost(params: str) -> str:
    """Calculate fine-tuning cost for a model.

    Args:
        params: JSON string with:
            - base_model: str (e.g., "gpt-4o-mini", "llama-3.1-8b")
            - training_tokens: int (total tokens in training data)
            - epochs: int (number of training passes)
            - num_runs: int (how many fine-tuning runs expected, including experimentation)
            - platform: str ("openai", "together", "self_hosted")

    Returns:
        Fine-tuning cost estimate
    """
    try:
        p = json.loads(params)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON params - {e}"

    base_model = p.get("base_model", "gpt-4o-mini")
    training_tokens = p.get("training_tokens", 1000000)
    epochs = p.get("epochs", 3)
    num_runs = p.get("num_runs", 3)
    platform = p.get("platform", "openai")

    # Rough fine-tuning prices per 1M training tokens
    ft_prices = {
        "openai": {"gpt-4o-mini": 3.0, "gpt-4o": 25.0, "gpt-3.5-turbo": 8.0},
        "together": {"default": 5.0},
        "self_hosted": {"default": 0},  # Only GPU cost
    }

    platform_prices = ft_prices.get(platform, ft_prices["together"])
    price_per_mtok = 0
    for key in [base_model, "default"]:
        if key in platform_prices:
            price_per_mtok = platform_prices[key]
            break

    total_tokens = training_tokens * epochs * num_runs
    total_cost = (total_tokens / 1_000_000) * price_per_mtok

    lines = [
        f"## Fine-tuning Cost Estimate\n",
        f"### Configuration",
        f"- Base model: {base_model}",
        f"- Platform: {platform}",
        f"- Training tokens: {training_tokens:,}",
        f"- Epochs: {epochs}",
        f"- Experimental runs: {num_runs}",
        f"- Total tokens processed: {total_tokens:,}",
        f"",
        f"### Cost",
        f"- Price: ${price_per_mtok:.2f} per 1M training tokens",
        f"- **Total fine-tuning cost: ${total_cost:.2f}**",
    ]

    if platform == "self_hosted":
        lines.append(f"\nNote: Self-hosted fine-tuning has no token cost but requires GPU rental.")
        lines.append(f"Use calculate_self_hosting_cost for GPU pricing.")

    return "\n".join(lines)


@tool
def calculate_scenario_costs(params: str) -> str:
    """Calculate costs for optimistic, realistic, and pessimistic scenarios.

    This is the KEY tool for producing cost RANGES (not fixed numbers).

    Args:
        params: JSON string with:
            - model_id: str
            - scenarios: dict with "optimistic", "realistic", "pessimistic" keys, each containing:
                - requests_per_day: int
                - avg_input_tokens: int
                - avg_output_tokens: int
                - conversation_turns: int
                - agent_calls_per_request: int
                - cache_hit_rate: float
                - batch_percentage: float

    Returns:
        Three-scenario cost comparison
    """
    try:
        p = json.loads(params)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON params - {e}"

    model_id = p.get("model_id", "")
    scenarios = p.get("scenarios", {})

    model = qdrant_manager.get_model(model_id)
    if not model:
        return f"Model '{model_id}' not found."

    input_price = model.get("input_price_per_mtok", 0)
    output_price = model.get("output_price_per_mtok", 0)
    model_name = model.get("name", model_id)

    if input_price == 0 and output_price == 0:
        return f"Model '{model_name}' has no API pricing. Use self-hosting cost tools."

    lines = [f"## Cost Scenarios: {model_name}\n"]
    lines.append(f"Pricing: ${input_price:.4f} / ${output_price:.4f} per 1M tokens (in/out)\n")

    for scenario_name in ["optimistic", "realistic", "pessimistic"]:
        s = scenarios.get(scenario_name, {})
        if not s:
            continue

        rpd = s.get("requests_per_day", 1000)
        avg_in = s.get("avg_input_tokens", 1000)
        avg_out = s.get("avg_output_tokens", 500)
        turns = s.get("conversation_turns", 1)
        agent_calls = s.get("agent_calls_per_request", 1)
        cache_rate = s.get("cache_hit_rate", 0)
        batch_pct = s.get("batch_percentage", 0)

        # Effective tokens
        if turns > 1:
            eff_input = (avg_in + (turns - 1) * (avg_in + avg_out) / 2) * agent_calls
        else:
            eff_input = avg_in * agent_calls
        eff_output = avg_out * agent_calls

        cost_per_req = (eff_input / 1e6) * input_price + (eff_output / 1e6) * output_price

        # Discounts
        regular_frac = max(0, 1 - cache_rate - batch_pct)
        daily = (rpd * regular_frac * cost_per_req +
                 rpd * cache_rate * ((eff_input / 1e6) * input_price * 0.5 + (eff_output / 1e6) * output_price) +
                 rpd * batch_pct * cost_per_req * 0.5)

        monthly = daily * 30
        yearly = daily * 365

        lines.append(f"### {scenario_name.title()}")
        lines.append(f"- Requests/day: {rpd:,} | Turns: {turns} | Agent calls: {agent_calls}")
        lines.append(f"- Input: {avg_in:,} tok | Output: {avg_out:,} tok")
        lines.append(f"- Cache: {cache_rate:.0%} | Batch: {batch_pct:.0%}")
        lines.append(f"- **Daily: ${daily:.2f} | Monthly: ${monthly:.2f} | Yearly: ${yearly:,.2f}**\n")

    return "\n".join(lines)


@tool
def generate_cost_table(params: str) -> str:
    """Generate a cost table at different volume levels for when exact volume is unknown.

    Args:
        params: JSON string with:
            - model_id: str
            - avg_input_tokens: int
            - avg_output_tokens: int
            - conversation_turns: int (default 1)
            - agent_calls_per_request: int (default 1)

    Returns:
        Cost table at [100, 500, 1000, 5000, 10000, 50000] requests/day
    """
    try:
        p = json.loads(params)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON params - {e}"

    model_id = p.get("model_id", "")
    avg_in = p.get("avg_input_tokens", 1000)
    avg_out = p.get("avg_output_tokens", 500)
    turns = p.get("conversation_turns", 1)
    agent_calls = p.get("agent_calls_per_request", 1)

    model = qdrant_manager.get_model(model_id)
    if not model:
        return f"Model '{model_id}' not found."

    input_price = model.get("input_price_per_mtok", 0)
    output_price = model.get("output_price_per_mtok", 0)
    model_name = model.get("name", model_id)

    if input_price == 0 and output_price == 0:
        return f"Model '{model_name}' has no API pricing."

    if turns > 1:
        eff_input = (avg_in + (turns - 1) * (avg_in + avg_out) / 2) * agent_calls
    else:
        eff_input = avg_in * agent_calls
    eff_output = avg_out * agent_calls

    cost_per_req = (eff_input / 1e6) * input_price + (eff_output / 1e6) * output_price

    volumes = [100, 500, 1000, 5000, 10000, 50000]

    lines = [
        f"## Cost Table: {model_name}\n",
        f"Assumptions: {avg_in:,} input + {avg_out:,} output tokens, {turns} turns, {agent_calls} agent calls\n",
        f"| Requests/Day | Daily | Monthly | Yearly |",
        f"|---|---|---|---|",
    ]

    for vol in volumes:
        daily = vol * cost_per_req
        monthly = daily * 30
        yearly = daily * 365
        lines.append(f"| {vol:,} | ${daily:.2f} | ${monthly:,.2f} | ${yearly:,.2f} |")

    lines.append(f"\nCost per request: ${cost_per_req:.6f}")

    return "\n".join(lines)


@tool
def validate_cost_result(result: str) -> str:
    """Validate a cost calculation result for sanity.

    Args:
        result: The cost result text to validate

    Returns:
        Validation result with any warnings
    """
    warnings = []

    # Extract dollar amounts from the text
    import re
    amounts = re.findall(r'\$([0-9,]+\.?\d*)', result)
    parsed_amounts = []
    for amt in amounts:
        try:
            parsed_amounts.append(float(amt.replace(",", "")))
        except ValueError:
            pass

    if not parsed_amounts:
        warnings.append("No dollar amounts found in the result.")

    for amt in parsed_amounts:
        if amt < 0:
            warnings.append(f"Negative cost found: ${amt}. Costs should never be negative.")
        if amt > 10_000_000:
            warnings.append(f"Extremely high cost: ${amt:,.2f}. Double-check your volume assumptions.")

    # Check for common issues
    lower = result.lower()
    if "daily" in lower and "monthly" in lower:
        # Try to verify monthly ≈ daily * 30
        pass  # Complex to parse, leave to the agent

    if not warnings:
        return "Validation passed: All costs look reasonable."

    return "Validation warnings:\n" + "\n".join(f"- {w}" for w in warnings)
