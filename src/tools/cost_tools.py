"""Tools for the Cost Agent: API cost, self-hosting, GPU options, embeddings, fine-tuning, scenarios, vector DB, dev costs, free tiers."""

import json
import os
import re
from langchain_core.tools import tool
from src.db import qdrant_manager

_gpu_data = None


def _load_gpu_data() -> dict:
    global _gpu_data
    if _gpu_data is None:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "gpu_pricing.json")
        with open(os.path.normpath(path)) as f:
            _gpu_data = json.load(f)
    return _gpu_data


def _calc_daily_api_cost(input_price, output_price, rpd, avg_in, avg_out, turns, agent_calls, cache_rate, batch_pct):
    """Shared helper: compute daily cost given pricing and usage parameters."""
    if turns > 1:
        avg_accumulated_input = avg_in + (turns - 1) * (avg_in + avg_out) / 2
        eff_input = avg_accumulated_input * agent_calls
    else:
        eff_input = avg_in * agent_calls
    eff_output = avg_out * agent_calls

    base_cost = (eff_input / 1e6) * input_price + (eff_output / 1e6) * output_price
    cached_cost = (eff_input / 1e6) * input_price * 0.5 + (eff_output / 1e6) * output_price
    batch_cost = base_cost * 0.5
    regular_frac = max(0.0, 1.0 - cache_rate - batch_pct)
    daily = (rpd * regular_frac * base_cost
             + rpd * cache_rate * cached_cost
             + rpd * batch_pct * batch_cost)
    return daily, eff_input, eff_output


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
            - batch_percentage: float (0-1, fraction using batch API — ~50% discount)
            - conversation_turns: int (average turns per conversation, for context accumulation)
            - agent_calls_per_request: int (if using agents, how many LLM calls per user request)
            - platform: str (optional — specific platform, e.g. "openai", "groq", "together")

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
    platform = p.get("platform", "")

    model = qdrant_manager.get_model(model_id)
    if not model:
        return (
            f"Model '{model_id}' not found in database. Cannot calculate cost without pricing data. "
            f"Please use search_models to find the correct model ID first."
        )

    # Resolve pricing: prefer platform-specific if requested
    input_price = 0.0
    output_price = 0.0
    pricing_note = ""

    if platform:
        pricing_array = model.get("pricing", [])
        for entry in pricing_array:
            if entry.get("platform", "").lower() == platform.lower():
                input_price = entry.get("input_price_per_mtok", 0)
                output_price = entry.get("output_price_per_mtok", 0)
                pricing_note = f" (platform: {platform})"
                break
        if not input_price and not output_price:
            pricing_note = f" (platform '{platform}' not found; using default pricing)"
            input_price = model.get("input_price_per_mtok", 0)
            output_price = model.get("output_price_per_mtok", 0)
    else:
        input_price = model.get("input_price_per_mtok", 0)
        output_price = model.get("output_price_per_mtok", 0)

    model_name = model.get("name", model_id)

    if input_price == 0 and output_price == 0:
        return (
            f"Model '{model_name}' has no API pricing (it's likely open-source/self-hosted). "
            f"Use calculate_self_hosting_cost instead."
        )

    daily, eff_input, eff_output = _calc_daily_api_cost(
        input_price, output_price, rpd, avg_input, avg_output, turns, agent_calls, cache_rate, batch_pct
    )

    # Add 12% production overhead (retries, monitoring, errors)
    overhead = daily * 0.12
    daily_with_overhead = daily + overhead
    monthly = daily_with_overhead * 30
    yearly = daily_with_overhead * 365
    base_cost_per_req = (eff_input / 1e6) * input_price + (eff_output / 1e6) * output_price

    lines = [
        f"## API Cost Estimate: {model_name}{pricing_note}\n",
        f"### Assumptions",
        f"- Input tokens/request: {avg_input:,} (effective with context accumulation & agent calls: {eff_input:,.0f})",
        f"- Output tokens/request: {avg_output:,} (effective: {eff_output:,.0f})",
        f"- Requests/day: {rpd:,}",
        f"- Conversation turns: {turns}",
        f"- Agent calls/request: {agent_calls}",
        f"- Cache hit rate: {cache_rate:.0%} (~50% input discount on cached tokens)",
        f"- Batch API usage: {batch_pct:.0%} (~50% discount)",
        f"",
        f"### Pricing{pricing_note}",
        f"- Input: ${input_price:.4f} per 1M tokens",
        f"- Output: ${output_price:.4f} per 1M tokens",
        f"",
        f"### Cost Breakdown",
        f"- Cost per request (base): ${base_cost_per_req:.6f}",
        f"- Base daily cost: ${daily:.2f}",
        f"- Production overhead (+12%): ${overhead:.2f}",
        f"- **Daily cost (with overhead): ${daily_with_overhead:.2f}**",
        f"- **Monthly cost (30d): ${monthly:.2f}**",
        f"- **Yearly cost (365d): ${yearly:.2f}**",
    ]

    return "\n".join(lines)


@tool
def calculate_self_hosting_cost(params: str) -> str:
    """Calculate the cost of self-hosting an AI model on rented GPUs.

    Args:
        params: JSON string with:
            - model_size: str (e.g., "7b", "13b", "70b", "405b")
            - quantization: str ("fp16", "int8", "int4")
            - gpu_name: str (e.g., "NVIDIA A100 80GB") — optional, auto-selects if omitted
            - provider: str (e.g., "runpod", "lambda_labs", "aws")
            - use_spot: bool (true = spot pricing if available, false = on-demand)
            - hours_per_day: int (24 for always-on, less for on-demand/scheduled)
            - redundancy: int (number of replicas for HA, minimum 1)
            - storage_gb: int (model weights + data storage needed, optional)
            - bandwidth_gb_per_month: int (estimated egress/ingress, optional)

    Returns:
        Self-hosting cost breakdown including compute, storage, bandwidth, and spot vs on-demand comparison
    """
    try:
        p = json.loads(params)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON params - {e}"

    model_size = p.get("model_size", "7b")
    quant = p.get("quantization", "fp16")
    gpu_name = p.get("gpu_name", "")
    provider = p.get("provider", "runpod")
    use_spot = p.get("use_spot", False)
    hours = p.get("hours_per_day", 24)
    redundancy = p.get("redundancy", 1)
    storage_gb = p.get("storage_gb", 0)
    bandwidth_gb = p.get("bandwidth_gb_per_month", 0)

    gpu_data = _load_gpu_data()
    vram_key = f"{model_size}_{quant}"
    vram_needed = gpu_data.get("model_vram_requirements", {}).get(vram_key, 0)

    if not vram_needed:
        return (
            f"Unknown model size/quantization: {model_size}/{quant}. "
            f"Valid options: {list(gpu_data['model_vram_requirements'].keys())}"
        )

    # Find GPU entry
    gpu_info = None
    for gpu in gpu_data["gpus"]:
        if gpu_name and gpu["name"].lower() == gpu_name.lower():
            gpu_info = gpu
            break
    if not gpu_info and not gpu_name:
        # Auto-select: cheapest GPU that fits VRAM and is available on this provider
        candidates = []
        for gpu in gpu_data["gpus"]:
            if gpu["vram_gb"] >= vram_needed and provider in gpu.get("providers", {}):
                pinfo = gpu["providers"][provider]
                price = pinfo.get("spot_usd") if use_spot and pinfo.get("spot_usd") else pinfo.get("on_demand_usd")
                if price:
                    candidates.append((gpu, price))
        if candidates:
            candidates.sort(key=lambda x: x[1])
            gpu_info = candidates[0][0]

    if not gpu_info:
        return (
            f"No suitable GPU found for {vram_needed}GB VRAM on provider '{provider}'. "
            f"Try a different provider or GPU."
        )

    gpus_needed = max(1, -(-vram_needed // gpu_info["vram_gb"]))  # ceiling division

    provider_info = gpu_info.get("providers", {}).get(provider)
    if not provider_info:
        available = list(gpu_info["providers"].keys())
        return f"GPU '{gpu_info['name']}' not available on '{provider}'. Available providers: {available}"

    on_demand_rate = provider_info.get("on_demand_usd")
    spot_rate = provider_info.get("spot_usd")

    if use_spot and spot_rate:
        hourly_rate = spot_rate
        pricing_type = "spot"
        spot_warning = (
            "\n⚠️  Spot instances can be interrupted with ~2 min notice. "
            "Not suitable for latency-sensitive production without fallback."
        )
    else:
        hourly_rate = on_demand_rate
        pricing_type = "on-demand"
        spot_warning = ""

    if not hourly_rate:
        return f"No {'spot' if use_spot else 'on-demand'} price found for {gpu_info['name']} on {provider}."

    total_gpus = gpus_needed * redundancy
    hourly_total = hourly_rate * total_gpus
    daily_compute = hourly_total * hours
    monthly_compute = daily_compute * 30

    # Storage cost estimate: ~$0.10/GB/month (rough cloud storage average)
    storage_cost_monthly = storage_gb * 0.10 if storage_gb else 0

    # Bandwidth/egress: ~$0.09/GB (AWS-ish baseline; providers vary)
    bandwidth_cost_monthly = bandwidth_gb * 0.09 if bandwidth_gb else 0

    total_monthly = monthly_compute + storage_cost_monthly + bandwidth_cost_monthly
    yearly = total_monthly * 12

    lines = [
        f"## Self-Hosting Cost Estimate ({pricing_type})\n",
        f"### Configuration",
        f"- Model: {model_size} ({quant}) — needs {vram_needed} GB VRAM",
        f"- GPU: {gpu_info['name']} ({gpu_info['vram_gb']} GB VRAM)",
        f"- GPUs per replica: {gpus_needed}",
        f"- Replicas (redundancy): {redundancy}",
        f"- Total GPUs: {total_gpus}",
        f"- Provider: {provider}",
        f"- Pricing type: {pricing_type}",
        f"- Hours/day running: {hours}",
        f"",
        f"### Pricing",
        f"- On-demand: ${on_demand_rate:.4f}/hr" if on_demand_rate else "- On-demand: not available",
        f"- Spot: ${spot_rate:.4f}/hr (~{round((1 - spot_rate/on_demand_rate)*100) if on_demand_rate and spot_rate else '?'}% cheaper)" if spot_rate else "- Spot: not available",
        f"- **Using: ${hourly_rate:.4f}/hr ({pricing_type})**",
        f"",
        f"### Monthly Cost Breakdown",
        f"- Compute: {total_gpus} GPU(s) × ${hourly_rate:.4f}/hr × {hours}h/day × 30d = ${monthly_compute:.2f}",
        f"- **Daily cost: ${daily_compute:.2f}**",
    ]
    if storage_gb:
        lines.append(f"- Storage ({storage_gb} GB × $0.10/GB/mo): ${storage_cost_monthly:.2f}")
    if bandwidth_gb:
        lines.append(f"- Bandwidth/egress ({bandwidth_gb} GB × $0.09/GB): ${bandwidth_cost_monthly:.2f}")
    lines += [
        f"- **Total monthly: ${total_monthly:.2f}**",
        f"- **Total yearly: ${yearly:.2f}**",
    ]

    # Show spot vs on-demand comparison when both available
    if on_demand_rate and spot_rate:
        od_monthly = on_demand_rate * total_gpus * hours * 30
        sp_monthly = spot_rate * total_gpus * hours * 30
        savings = od_monthly - sp_monthly
        lines += [
            f"",
            f"### Spot vs On-Demand Comparison",
            f"- On-demand monthly: ${od_monthly:.2f}",
            f"- Spot monthly: ${sp_monthly:.2f}",
            f"- Spot savings: ${savings:.2f}/mo ({round(savings/od_monthly*100)}% cheaper)",
            f"- Spot risk: ~2 min interruption notice — plan checkpointing/fallback accordingly",
        ]
    elif not spot_rate:
        lines.append(f"\nNote: Spot pricing not available for {gpu_info['name']} on {provider}.")

    if spot_warning:
        lines.append(spot_warning)

    lines += [
        f"",
        f"### Additional Notes",
        f"- DevOps overhead for self-hosting: monitoring, deployment, scaling are NOT included in cost above",
        f"- Cold start time: model loading can take 30s–5min depending on model size",
        f"- Idle cost: if running {hours}h/day, paying for idle time between requests",
        f"- Storage above assumes ~${storage_gb * 0.10:.2f}/mo; adjust with actual model weights size",
    ]

    return "\n".join(lines)


@tool
def get_gpu_options(model_id: str) -> str:
    """Get GPU options that can run a given model, with on-demand AND spot pricing across providers.

    Args:
        model_id: Model identifier or size descriptor (e.g., "70b", "meta-llama/llama-3.1-70b")

    Returns:
        Table of GPU options with on-demand and spot pricing per provider
    """
    gpu_data = _load_gpu_data()

    size_str = model_id.lower()
    vram_needed = None

    for size_key, vram in gpu_data["model_vram_requirements"].items():
        size_part = size_key.split("_")[0]
        if size_part in size_str:
            full_key = f"{size_part}_int4"
            vram_needed = gpu_data["model_vram_requirements"].get(full_key, vram)
            break

    if not vram_needed:
        vram_needed = 16  # conservative default for unknown models

    lines = [
        f"## GPU Options for Model: {model_id}",
        f"Estimated VRAM needed: ~{vram_needed} GB (INT4 quantization)\n",
        f"On-demand = reliable, any time. Spot = interruptible (~70% cheaper, 2-min notice).\n",
    ]

    for gpu in gpu_data["gpus"]:
        if gpu["vram_gb"] < vram_needed:
            continue

        gpus_needed = max(1, -(-vram_needed // gpu["vram_gb"]))
        lines.append(f"### {gpu['name']} ({gpu['vram_gb']}GB VRAM, {gpus_needed} GPU(s) needed)")

        for provider, info in gpu["providers"].items():
            on_demand = info.get("on_demand_usd")
            spot = info.get("spot_usd")

            if on_demand:
                od_hr = on_demand * gpus_needed
                od_mo = od_hr * 24 * 30
                od_str = f"${od_hr:.2f}/hr (${od_mo:,.0f}/mo always-on)"
            else:
                od_str = "n/a"

            if spot:
                sp_hr = spot * gpus_needed
                sp_mo = sp_hr * 24 * 30
                savings_pct = round((1 - spot / on_demand) * 100) if on_demand else "?"
                sp_str = f"${sp_hr:.2f}/hr (${sp_mo:,.0f}/mo, ~{savings_pct}% cheaper than on-demand)"
            else:
                sp_str = "not available"

            lines.append(f"  - **{provider}**: on-demand {od_str} | spot {sp_str}")

        lines.append("")

    return "\n".join(lines)


@tool
def calculate_embedding_cost(params: str) -> str:
    """Calculate embedding cost for RAG pipeline.

    Args:
        params: JSON string with:
            - embedding_model_id: str (e.g., "openai/text-embedding-3-small") — must be in DB
            - num_documents: int
            - avg_tokens_per_doc: int
            - re_embedding_frequency: str ("once", "weekly", "monthly")
            - query_tokens_per_day: int (tokens for query embeddings per day)

    Returns:
        Embedding cost breakdown. Returns an error if model not found or has no pricing.
    """
    try:
        p = json.loads(params)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON params - {e}"

    model_id = p.get("embedding_model_id", "")
    num_docs = p.get("num_documents", 1000)
    avg_tokens = p.get("avg_tokens_per_doc", 500)
    re_embed = p.get("re_embedding_frequency", "once")
    query_tokens_day = p.get("query_tokens_per_day", 100000)

    if not model_id:
        # Find embedding models in DB to suggest actual options with real prices
        candidates = qdrant_manager.semantic_search("embedding model text vectorization", top_k=5,
                                                     filters={"type": "embedding"})
        if not candidates:
            candidates = qdrant_manager.semantic_search("text embedding model", top_k=5)
        suggestions = [
            f"{m.get('id', m.get('name', ''))} (${m.get('input_price_per_mtok', 0):.4f}/1M tokens)"
            for m in candidates if m.get("input_price_per_mtok", 0) > 0
        ]
        return (
            "Error: embedding_model_id is required. "
            + (f"Embedding models available in DB: {suggestions}. " if suggestions else "")
            + "Use search_models to find embedding models, then retry with the exact model ID."
        )

    model = qdrant_manager.get_model(model_id)
    if not model:
        # Try a fuzzy search for embedding models in DB
        similar = qdrant_manager.semantic_search(f"embedding model similar to {model_id}", top_k=3)
        suggestions = [m.get("id", m.get("name", "")) for m in similar] if similar else []
        return (
            f"Embedding model '{model_id}' not found in database. "
            f"Cannot calculate cost without pricing data. "
            + (f"Similar models in DB: {suggestions}. " if suggestions else "")
            + "Use search_models to find the right embedding model, then retry with its exact ID."
        )

    price_per_mtok = model.get("input_price_per_mtok", 0)
    if not price_per_mtok:
        return (
            f"Model '{model_id}' found but has no pricing data. "
            f"This may be a self-hosted/free model (e.g., sentence-transformers). "
            f"If self-hosted, embedding cost is effectively $0 for tokens — only infrastructure cost applies. "
            f"Use calculate_self_hosting_cost for infrastructure."
        )

    model_name = model.get("name", model_id)

    total_doc_tokens = num_docs * avg_tokens
    initial_embed_cost = (total_doc_tokens / 1_000_000) * price_per_mtok

    re_embed_monthly = 0.0
    re_embed_label = "None (one-time only)"
    if re_embed == "weekly":
        re_embed_monthly = initial_embed_cost * 4
        re_embed_label = "4× initial cost/month"
    elif re_embed == "monthly":
        re_embed_monthly = initial_embed_cost
        re_embed_label = "1× initial cost/month"

    query_daily_cost = (query_tokens_day / 1_000_000) * price_per_mtok
    query_monthly_cost = query_daily_cost * 30
    total_monthly = re_embed_monthly + query_monthly_cost
    total_yearly = total_monthly * 12

    lines = [
        f"## Embedding Cost Estimate\n",
        f"### Configuration",
        f"- Model: {model_name}",
        f"- Price: ${price_per_mtok:.4f} per 1M tokens",
        f"- Documents: {num_docs:,} × {avg_tokens:,} tokens = {total_doc_tokens:,} total tokens",
        f"- Re-embedding frequency: {re_embed} ({re_embed_label})",
        f"- Query volume: {query_tokens_day:,} tokens/day",
        f"",
        f"### Cost Breakdown",
        f"- **Initial embedding (one-time): ${initial_embed_cost:.4f}**",
        f"- Re-embedding cost/month: ${re_embed_monthly:.4f}",
        f"- Query embedding cost/month: ${query_monthly_cost:.4f}",
        f"- **Total ongoing monthly: ${total_monthly:.4f}**",
        f"- **Total yearly (ongoing): ${total_yearly:.2f}**",
        f"",
        f"Note: Initial cost is one-time. Monthly = re-embedding + queries.",
    ]

    return "\n".join(lines)


@tool
def calculate_finetuning_cost(params: str) -> str:
    """Calculate fine-tuning cost for a model.

    Args:
        params: JSON string with:
            - base_model: str (model ID to look up in DB for training price)
            - training_tokens: int (total tokens in training dataset)
            - epochs: int (number of training passes)
            - num_runs: int (how many fine-tuning runs including experimentation, typically 3-5)
            - platform: str ("api" for managed provider, "self_hosted" for GPU rental)

    Returns:
        Fine-tuning cost estimate using real pricing from DB (via LiteLLM source).
        Returns explicit error + guidance if pricing not found.
    """
    try:
        p = json.loads(params)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON params - {e}"

    base_model = p.get("base_model", "")
    training_tokens = p.get("training_tokens", 1_000_000)
    epochs = p.get("epochs", 3)
    num_runs = p.get("num_runs", 3)
    platform = p.get("platform", "api")

    if platform == "self_hosted":
        return (
            f"## Fine-tuning Cost: Self-Hosted\n"
            f"Self-hosted fine-tuning has no per-token training cost — cost is GPU rental time.\n"
            f"Training time depends on model size, dataset, and GPU. Rough rule of thumb:\n"
            f"- 7B model, 1M tokens, 3 epochs ≈ 2-8 hours on A100 80GB\n"
            f"- Use calculate_self_hosting_cost with hours_per_day for training duration\n"
            f"- Then multiply by num_runs ({num_runs}) for total experimentation cost\n"
            f"Also factor in: dataset preparation time, evaluation runs (separate inference cost)."
        )

    if not base_model:
        return "Error: base_model is required to look up fine-tuning pricing."

    model = qdrant_manager.get_model(base_model)
    training_cost_per_mtok = None
    model_name = base_model

    if model:
        model_name = model.get("name", base_model)
        raw = model.get("training_cost_per_mtok", 0)
        if raw and raw > 0:
            training_cost_per_mtok = raw

    if training_cost_per_mtok is None:
        # Try to search for the model with fine-tuning capability
        return (
            f"## Fine-tuning Cost: No Data Available\n"
            f"No fine-tuning pricing found for '{base_model}' in the database.\n"
            f"\nThis means either:\n"
            f"1. This model does not support managed fine-tuning via API\n"
            f"2. Pricing is not yet in LiteLLM's pricing database\n"
            f"\nWhat to do:\n"
            f"- Check the provider's fine-tuning page directly for current pricing\n"
            f"- For OpenAI models: platform.openai.com/docs/guides/fine-tuning\n"
            f"- For Together AI: docs.together.ai/docs/fine-tuning\n"
            f"- For Fireworks AI: docs.fireworks.ai/fine-tuning\n"
            f"\nFor self-hosted fine-tuning on GPU, use platform='self_hosted' instead."
        )

    total_tokens = training_tokens * epochs * num_runs
    total_cost = (total_tokens / 1_000_000) * training_cost_per_mtok

    # Per-run cost breakdown
    tokens_per_run = training_tokens * epochs
    cost_per_run = (tokens_per_run / 1_000_000) * training_cost_per_mtok

    lines = [
        f"## Fine-tuning Cost Estimate: {model_name}\n",
        f"### Configuration",
        f"- Base model: {model_name}",
        f"- Training tokens (dataset size): {training_tokens:,}",
        f"- Epochs: {epochs}",
        f"- Experimental runs: {num_runs} (iterations to get right)",
        f"- Total tokens billed: {training_tokens:,} × {epochs} epochs × {num_runs} runs = {total_tokens:,}",
        f"",
        f"### Pricing",
        f"- Training price: ${training_cost_per_mtok:.4f} per 1M tokens (from LiteLLM pricing data)",
        f"",
        f"### Cost Breakdown",
        f"- Cost per run ({epochs} epochs): ${cost_per_run:.2f}",
        f"- **Total for {num_runs} runs: ${total_cost:.2f}**",
        f"",
        f"### Additional Costs to Factor In",
        f"- Dataset preparation (if using synthetic data generation): separate inference cost",
        f"- Evaluation/testing inference cost after each run",
        f"- Periodic retraining as data drifts (multiply cost by expected retraining frequency/year)",
        f"- Inference on the fine-tuned model (may differ from base model price — check provider)",
    ]

    return "\n".join(lines)


@tool
def calculate_vector_db_cost(params: str) -> str:
    """Estimate vector database storage size and guide the user to check current pricing.

    This tool does NOT have hardcoded pricing — vector DB providers don't expose pricing via API
    and prices change frequently. It calculates your storage/traffic requirements so you can
    look up the cost yourself at the provider's pricing page.

    Args:
        params: JSON string with:
            - provider: str — one of: "qdrant_cloud", "pinecone", "weaviate", "milvus_zilliz",
                                       "chroma", "pgvector" (self-hosted), "qdrant" (self-hosted)
            - num_vectors: int (total number of vectors to store)
            - vector_dim: int (embedding dimensions, e.g., 384 for MiniLM, 1536 for OpenAI ada)
            - reads_per_month: int (search queries per month)
            - writes_per_month: int (upsert/insert operations per month)

    Returns:
        Storage size calculation + pricing page URL + what cost factors to look for
    """
    try:
        p = json.loads(params)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON params - {e}"

    provider = p.get("provider", "")
    num_vectors = p.get("num_vectors", 100_000)
    vector_dim = p.get("vector_dim", 768)
    reads_per_month = p.get("reads_per_month", 100_000)
    writes_per_month = p.get("writes_per_month", 10_000)

    # Storage math — this part IS calculable (pure math, no pricing needed)
    raw_storage_gb = (num_vectors * vector_dim * 4) / 1e9  # float32 = 4 bytes
    with_index_gb = raw_storage_gb * 2.0  # typical index overhead ~2x

    # Pricing page URLs and what pricing model each uses
    provider_info = {
        "qdrant_cloud": {
            "url": "https://qdrant.tech/pricing/",
            "model": "node-based (per node/hour) + free tier (1 cluster, 1GB)",
            "self_hosted": False,
        },
        "pinecone": {
            "url": "https://www.pinecone.io/pricing/",
            "model": "serverless: per read unit + per write unit + per GB storage; or pod-based (per pod/hour)",
            "self_hosted": False,
        },
        "weaviate": {
            "url": "https://weaviate.io/pricing",
            "model": "node-based (per node/hour) + free sandbox (14-day limit)",
            "self_hosted": False,
        },
        "milvus_zilliz": {
            "url": "https://zilliz.com/pricing",
            "model": "compute unit based (per CU/hour) + free tier (1 cluster)",
            "self_hosted": False,
        },
        "chroma": {
            "url": "https://www.trychroma.com/",
            "model": "self-hosted only — no SaaS, cost = your server infrastructure",
            "self_hosted": True,
        },
        "pgvector": {
            "url": "https://github.com/pgvector/pgvector",
            "model": "self-hosted only — runs inside PostgreSQL, cost = your existing DB infrastructure",
            "self_hosted": True,
        },
        "qdrant": {
            "url": "https://github.com/qdrant/qdrant",
            "model": "self-hosted only — cost = your server infrastructure",
            "self_hosted": True,
        },
    }

    info = provider_info.get(provider)
    if not info:
        available = list(provider_info.keys())
        return (
            f"Provider '{provider}' not recognized. Available options: {available}\n"
            f"If your provider isn't listed, check their pricing page directly."
        )

    lines = [
        f"## Vector DB Requirements: {provider}\n",
        f"### Your Storage Requirements (calculated)",
        f"- Vectors: {num_vectors:,} × {vector_dim} dimensions",
        f"- Raw vector storage: {raw_storage_gb:.3f} GB ({num_vectors:,} × {vector_dim} × 4 bytes)",
        f"- With index overhead (~2×): ~{with_index_gb:.2f} GB",
        f"- Monthly reads: {reads_per_month:,}",
        f"- Monthly writes: {writes_per_month:,}",
        f"",
        f"### Pricing (check current rates — I don't store these)",
        f"- Pricing model: {info['model']}",
        f"- Current pricing page: {info['url']}",
        f"",
    ]

    if info["self_hosted"]:
        lines += [
            f"**Self-hosted: no SaaS cost.** Cost = infrastructure only.",
            f"- Use calculate_self_hosting_cost to estimate server/GPU cost",
            f"- Or factor into your existing server cost if running alongside other services",
        ]
    else:
        lines += [
            f"### What to look up at {info['url']}",
            f"- Does your {with_index_gb:.1f} GB fit in the free tier?",
            f"- For paid tier: what is the per-node/hour rate (or per-unit if usage-based)?",
            f"- If usage-based (like Pinecone serverless): note the per-read and per-write costs",
            f"  → Your monthly read cost = reads_per_month × price_per_read",
            f"  → Your monthly write cost = writes_per_month × price_per_write",
            f"  → Your storage cost = {with_index_gb:.2f} GB × price_per_gb_per_month",
            f"",
            f"Once you have the rates, tell me and I'll calculate the exact monthly cost.",
        ]

    return "\n".join(lines)


@tool
def estimate_development_costs(params: str) -> str:
    """Estimate pre-production development and testing costs before a system goes live.

    Args:
        params: JSON string with:
            - project_complexity: str ("simple", "moderate", "complex")
              simple = single model, few prompts, no RAG/fine-tuning
              moderate = multi-model, RAG, or moderate agent pipeline
              complex = multi-agent, fine-tuning, real-time requirements, compliance
            - primary_model_id: str (optional, model used for dev/testing)
            - expected_prod_requests_per_day: int (helps scale dev testing estimate)
            - include_synthetic_data_gen: bool (whether synthetic data generation is needed)

    Returns:
        Development and testing cost range before production launch
    """
    try:
        p = json.loads(params)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON params - {e}"

    complexity = p.get("project_complexity", "moderate")
    model_id = p.get("primary_model_id", "")
    prod_rpd = p.get("expected_prod_requests_per_day", 1000)
    include_synth = p.get("include_synthetic_data_gen", False)

    # Multipliers: ratio of dev testing volume to 1 month of prod (rough heuristic)
    multipliers = {
        "simple":   {"min": 0.5, "max": 2.0,  "weeks": "2-4"},
        "moderate": {"min": 1.5, "max": 5.0,  "weeks": "4-8"},
        "complex":  {"min": 3.0, "max": 10.0, "weeks": "8-16"},
    }
    m = multipliers.get(complexity, multipliers["moderate"])

    # Estimate testing volume as a fraction of one prod month
    test_requests_low = int(prod_rpd * 30 * m["min"] * 0.1)   # 10% of prod scale
    test_requests_high = int(prod_rpd * 30 * m["max"] * 0.15)  # 15% of prod scale

    lines = [
        f"## Development & Testing Cost Estimate\n",
        f"### Complexity: {complexity.title()} ({m['weeks']} weeks typical development)",
        f"",
        f"Before going live, every system incurs development and testing costs:\n",
        f"**What drives dev/test cost:**",
        f"- Prompt engineering iterations (many small test calls)",
        f"- Integration testing (end-to-end runs at moderate volume)",
        f"- Evaluation runs (testing model quality across test sets)",
        f"- Bug fixing cycles (often repeats earlier phases)",
        f"- Load/performance testing (simulating prod load)",
        f"",
        f"### Estimated Testing Volume",
        f"- Estimated test requests over dev period: {test_requests_low:,} – {test_requests_high:,}",
        f"- This is roughly {m['min']*10:.0f}–{m['max']*15:.0f}% of one production month",
        f"",
    ]

    if model_id:
        model = qdrant_manager.get_model(model_id)
        if model:
            input_price = model.get("input_price_per_mtok", 0)
            output_price = model.get("output_price_per_mtok", 0)
            if input_price or output_price:
                avg_in, avg_out = 500, 300
                cost_per_req = (avg_in / 1e6) * input_price + (avg_out / 1e6) * output_price
                dev_cost_low = test_requests_low * cost_per_req
                dev_cost_high = test_requests_high * cost_per_req
                lines += [
                    f"### Estimated Dev/Test API Cost ({model.get('name', model_id)})",
                    f"- Cost per test request (500 in + 300 out tokens): ${cost_per_req:.6f}",
                    f"- **Estimated dev/test API cost: ${dev_cost_low:.2f} – ${dev_cost_high:.2f}**",
                    f"",
                ]

    lines += [
        f"### Additional Dev Costs (non-API)",
        f"- Evaluation framework setup and manual review time",
        f"- Staging environment / infra cost (if self-hosting: multiply by {m['weeks'].split('-')[0]} weeks × infra cost)",
    ]

    if include_synth:
        lines += [
            f"",
            f"### Synthetic Data Generation",
            f"- You've indicated synthetic data generation is needed",
            f"- This requires running a capable LLM (often GPT-4 class) on your full dataset",
            f"- Use calculate_api_cost with the data generation model for this estimate",
            f"- Typical: 1000 examples × 2000 tokens output = 2M tokens of output",
        ]

    lines += [
        f"",
        f"Note: These are rough estimates. Actual dev cost depends heavily on team experience, "
        f"prompt complexity, and how many evaluation/fix cycles are needed.",
    ]

    return "\n".join(lines)


@tool
def get_free_tier_info(provider: str) -> str:
    """Get the free tier / pricing page URL for an AI provider so the user can check current limits.

    This tool does NOT store free tier data — free tier limits and credits change frequently
    and providers don't expose them via API. It directs you to the right page to check.

    Args:
        provider: str (e.g., "groq", "openrouter", "together", "anthropic", "openai",
                        "cohere", "fireworks", "cerebras", "huggingface", "replicate",
                        "mistral", "deepinfra", "perplexity")

    Returns:
        Pricing/free tier page URL + what to look for
    """
    # These are stable documentation URLs, not pricing data
    provider_pages = {
        "groq":        ("https://console.groq.com/settings/limits",
                        "Check 'Limits' in console — shows RPM/TPM per model. Free tier, no credit card."),
        "openrouter":  ("https://openrouter.ai/models",
                        "Filter by '$0' to see permanently free models. Signup also gives $1 credit."),
        "together":    ("https://www.together.ai/pricing",
                        "Check current trial credits at signup and per-token rates."),
        "anthropic":   ("https://www.anthropic.com/pricing",
                        "Check API pricing page. Trial credits given at signup — amount varies."),
        "openai":      ("https://platform.openai.com/docs/pricing",
                        "Check current rates. Trial credits at signup — amount and expiry vary."),
        "cohere":      ("https://cohere.com/pricing",
                        "Trial API key has rate limits — check current limits on their pricing page."),
        "fireworks":   ("https://fireworks.ai/pricing",
                        "Check current trial credits and per-token rates."),
        "cerebras":    ("https://cerebras.ai/inference",
                        "Check free tier limits — known for high TPM free tier but limits change."),
        "huggingface": ("https://huggingface.co/docs/api-inference/en/rate-limits",
                        "Free Inference API for public models with rate limits — check current limits."),
        "replicate":   ("https://replicate.com/pricing",
                        "Pay-per-second for compute. No free tier, but small free credit at signup."),
        "mistral":     ("https://mistral.ai/technology/#pricing",
                        "Check current per-token rates and any free trial credits."),
        "deepinfra":   ("https://deepinfra.com/pricing",
                        "Pay-per-token. Check current rates and any signup credits."),
        "perplexity":  ("https://www.perplexity.ai/hub/blog/introducing-pplx-api",
                        "Check current API pricing and free tier."),
    }

    entry = provider_pages.get(provider.lower())
    if not entry:
        available = sorted(provider_pages.keys())
        return (
            f"Provider '{provider}' not in my list. Known providers: {available}\n"
            f"For any other provider, search '[provider name] API pricing' or check their docs."
        )

    url, guidance = entry
    return (
        f"## Free Tier / Pricing: {provider}\n"
        f"\n"
        f"**Pricing page:** {url}\n"
        f"**What to check:** {guidance}\n"
        f"\n"
        f"⚠️  Free tier limits and credits change frequently — I don't store these because "
        f"stale data is worse than no data. Always verify at the link above before planning around free tiers."
    )


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
                - label: str (optional — name this scenario, e.g. "low usage, good caching")
            - platform: str (optional — specific platform)

    Returns:
        Three-scenario cost comparison
    """
    try:
        p = json.loads(params)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON params - {e}"

    model_id = p.get("model_id", "")
    scenarios = p.get("scenarios", {})
    platform = p.get("platform", "")

    model = qdrant_manager.get_model(model_id)
    if not model:
        return f"Model '{model_id}' not found."

    input_price = 0.0
    output_price = 0.0
    pricing_note = ""

    if platform:
        for entry in model.get("pricing", []):
            if entry.get("platform", "").lower() == platform.lower():
                input_price = entry.get("input_price_per_mtok", 0)
                output_price = entry.get("output_price_per_mtok", 0)
                pricing_note = f" on {platform}"
                break
    if not input_price and not output_price:
        input_price = model.get("input_price_per_mtok", 0)
        output_price = model.get("output_price_per_mtok", 0)

    model_name = model.get("name", model_id)

    if input_price == 0 and output_price == 0:
        return f"Model '{model_name}' has no API pricing. Use self-hosting cost tools."

    lines = [f"## Cost Scenarios: {model_name}{pricing_note}\n"]
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
        label = s.get("label", "")

        daily, eff_input, eff_output = _calc_daily_api_cost(
            input_price, output_price, rpd, avg_in, avg_out, turns, agent_calls, cache_rate, batch_pct
        )
        overhead = daily * 0.12
        daily_total = daily + overhead
        monthly = daily_total * 30
        yearly = daily_total * 365

        header = f"### {scenario_name.title()}"
        if label:
            header += f" — {label}"
        lines.append(header)
        lines.append(f"- Requests/day: {rpd:,} | Turns: {turns} | Agent calls: {agent_calls}")
        lines.append(f"- Input: {avg_in:,} tok | Output: {avg_out:,} tok")
        lines.append(f"- Cache: {cache_rate:.0%} | Batch: {batch_pct:.0%}")
        lines.append(f"- Overhead: +12% for retries/monitoring")
        lines.append(f"- **Daily: ${daily_total:.2f} | Monthly: ${monthly:.2f} | Yearly: ${yearly:,.2f}**\n")

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
            - platform: str (optional)

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
    platform = p.get("platform", "")

    model = qdrant_manager.get_model(model_id)
    if not model:
        return f"Model '{model_id}' not found."

    input_price = 0.0
    output_price = 0.0
    pricing_note = ""

    if platform:
        for entry in model.get("pricing", []):
            if entry.get("platform", "").lower() == platform.lower():
                input_price = entry.get("input_price_per_mtok", 0)
                output_price = entry.get("output_price_per_mtok", 0)
                pricing_note = f" ({platform})"
                break
    if not input_price and not output_price:
        input_price = model.get("input_price_per_mtok", 0)
        output_price = model.get("output_price_per_mtok", 0)

    model_name = model.get("name", model_id)

    if input_price == 0 and output_price == 0:
        return f"Model '{model_name}' has no API pricing."

    # Calculate effective tokens (no cache/batch — baseline)
    if turns > 1:
        eff_input = (avg_in + (turns - 1) * (avg_in + avg_out) / 2) * agent_calls
    else:
        eff_input = avg_in * agent_calls
    eff_output = avg_out * agent_calls

    cost_per_req = (eff_input / 1e6) * input_price + (eff_output / 1e6) * output_price
    cost_with_overhead = cost_per_req * 1.12  # +12% production overhead

    volumes = [100, 500, 1000, 5000, 10000, 50000]

    lines = [
        f"## Cost Table: {model_name}{pricing_note}\n",
        f"Assumptions: {avg_in:,} input + {avg_out:,} output tokens, {turns} turns, {agent_calls} agent calls, +12% overhead\n",
        f"| Requests/Day | Daily | Monthly | Yearly |",
        f"|---|---|---|---|",
    ]

    for vol in volumes:
        daily = vol * cost_with_overhead
        monthly = daily * 30
        yearly = daily * 365
        lines.append(f"| {vol:,} | ${daily:.2f} | ${monthly:,.2f} | ${yearly:,.2f} |")

    lines.append(f"\nBase cost per request: ${cost_per_req:.6f} | With overhead: ${cost_with_overhead:.6f}")

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

    if not warnings:
        return "Validation passed: All costs look reasonable."

    return "Validation warnings:\n" + "\n".join(f"- {w}" for w in warnings)
