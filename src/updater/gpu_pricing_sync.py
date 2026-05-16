"""GPU pricing sync from SkyPilot Catalog.

Source: https://github.com/skypilot-org/skypilot/tree/master/sky/clouds/catalog/data
- No auth required (public GitHub raw CSVs)
- Updated every ~7 hours by SkyPilot CI
- Covers AWS, GCP, Azure, Lambda Labs, RunPod

On failure: keeps existing JSON values unchanged (silent fallback).
Stores both on-demand and spot pricing per provider per GPU.
"""

import csv
import io
import json
import os

import requests

GPU_PRICING_PATH = "config/gpu_pricing.json"

# SkyPilot catalog CSV URLs — catalog moved to separate repo: skypilot-org/skypilot-catalog
_SKYPILOT_URLS = {
    "aws": "https://raw.githubusercontent.com/skypilot-org/skypilot-catalog/master/catalogs/v8/aws/vms.csv",
    "gcp": "https://raw.githubusercontent.com/skypilot-org/skypilot-catalog/master/catalogs/v8/gcp/vms.csv",
    "azure": "https://raw.githubusercontent.com/skypilot-org/skypilot-catalog/master/catalogs/v8/azure/vms.csv",
    "lambda_labs": "https://raw.githubusercontent.com/skypilot-org/skypilot-catalog/master/catalogs/v8/lambda/vms.csv",
    "runpod": "https://raw.githubusercontent.com/skypilot-org/skypilot-catalog/master/catalogs/v8/runpod/vms.csv",
}

# Map SkyPilot AcceleratorName → our GPU name in gpu_pricing.json
_GPU_NAME_MAP = {
    "T4": "NVIDIA T4",
    "L4": "NVIDIA L4",
    "A10G": "NVIDIA A10G",
    "A10": "NVIDIA A10G",
    "L40S": "NVIDIA L40S",
    "L40s": "NVIDIA L40S",
    "A100-40GB": "NVIDIA A100 40GB",
    "A100": "NVIDIA A100 40GB",
    "A100-80GB": "NVIDIA A100 80GB",
    "A100-80SXM": "NVIDIA A100 80GB",
    "A100-SXM4-80GB": "NVIDIA A100 80GB",
    "H100": "NVIDIA H100 80GB",
    "H100-80GB": "NVIDIA H100 80GB",
    "H100-SXM": "NVIDIA H100 80GB",
    "H100-SXM5-80GB": "NVIDIA H100 80GB",
    "H200": "NVIDIA H200",
    "H200-141GB": "NVIDIA H200",
    "H200-SXM": "NVIDIA H200",
}


def _fetch_provider_prices(provider: str, url: str) -> dict:
    """Fetch CSV and return {our_gpu_name: {"on_demand_usd": float|None, "spot_usd": float|None, "instance": str}}.

    For each GPU type, picks the cheapest on-demand instance AND the cheapest spot instance separately.
    Returns {} on any failure (caller keeps existing values).
    """
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [WARN] gpu_pricing_sync: could not reach {provider}: {e}")
        return {}

    # best_ondemand[gpu_name] = {"price": float, "instance": str}
    # best_spot[gpu_name]     = {"price": float, "instance": str}
    best_ondemand: dict[str, dict] = {}
    best_spot: dict[str, dict] = {}

    try:
        reader = csv.DictReader(io.StringIO(resp.text))
        for row in reader:
            acc_name = (row.get("AcceleratorName") or "").strip()
            acc_count_raw = (row.get("AcceleratorCount") or "0").strip()
            price_raw = (row.get("Price") or "").strip()
            spot_raw = (row.get("SpotPrice") or "").strip()
            instance = (row.get("InstanceType") or "").strip()

            if not acc_name or acc_name.lower() in ("", "nan", "none"):
                continue

            # Only single-GPU instances (count may be float like "1.0")
            try:
                if float(acc_count_raw) != 1.0:
                    continue
            except (ValueError, TypeError):
                continue

            our_gpu = _GPU_NAME_MAP.get(acc_name)
            if not our_gpu:
                continue

            # On-demand price
            try:
                price = float(price_raw)
                if price > 0:
                    if our_gpu not in best_ondemand or price < best_ondemand[our_gpu]["price"]:
                        best_ondemand[our_gpu] = {"price": round(price, 4), "instance": instance}
            except (ValueError, TypeError):
                pass

            # Spot price (may not exist for all providers/rows)
            try:
                spot = float(spot_raw)
                if spot > 0:
                    if our_gpu not in best_spot or spot < best_spot[our_gpu]["price"]:
                        best_spot[our_gpu] = {"price": round(spot, 4), "instance": instance}
            except (ValueError, TypeError):
                pass

    except Exception as e:
        print(f"  [WARN] gpu_pricing_sync: failed to parse {provider} CSV: {e}")
        return {}

    # Merge into single result per GPU
    all_gpus = set(best_ondemand) | set(best_spot)
    result = {}
    for gpu in all_gpus:
        on_demand = best_ondemand.get(gpu)
        spot = best_spot.get(gpu)
        result[gpu] = {
            "on_demand_usd": on_demand["price"] if on_demand else None,
            "spot_usd": spot["price"] if spot else None,
            "instance": (on_demand or spot or {}).get("instance", ""),
        }

    return result


def update_gpu_pricing() -> bool:
    """Fetch fresh GPU prices from SkyPilot Catalog and update gpu_pricing.json.

    Updates both on_demand_usd and spot_usd per provider per GPU.
    Returns True if at least one price was updated, False otherwise.
    On any per-provider failure, existing JSON values are kept unchanged.
    """
    print("Fetching GPU pricing from SkyPilot Catalog...")

    # Load existing JSON
    try:
        with open(GPU_PRICING_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [WARN] gpu_pricing_sync: could not load {GPU_PRICING_PATH}: {e}")
        return False

    # Fetch fresh prices per provider
    fresh: dict[str, dict] = {}  # provider -> {gpu_name -> {on_demand_usd, spot_usd, instance}}
    for provider, url in _SKYPILOT_URLS.items():
        prices = _fetch_provider_prices(provider, url)
        if prices:
            fresh[provider] = prices
            print(f"  {provider}: fetched {len(prices)} GPU prices")
        else:
            print(f"  {provider}: fetch failed, keeping existing prices")

    if not fresh:
        print("  No providers updated — keeping existing gpu_pricing.json unchanged")
        return False

    # Merge into existing JSON
    updated_count = 0
    for gpu_entry in data.get("gpus", []):
        gpu_name = gpu_entry.get("name", "")
        providers_dict = gpu_entry.get("providers", {})
        for provider, provider_entry in providers_dict.items():
            if provider not in fresh:
                continue
            new_data = fresh[provider].get(gpu_name)
            if not new_data:
                continue

            changed = False

            # Update on_demand_usd
            if new_data["on_demand_usd"] is not None:
                if provider_entry.get("on_demand_usd") != new_data["on_demand_usd"]:
                    provider_entry["on_demand_usd"] = new_data["on_demand_usd"]
                    changed = True

            # Update spot_usd (can be set to None if not available)
            new_spot = new_data["spot_usd"]
            if provider_entry.get("spot_usd") != new_spot:
                provider_entry["spot_usd"] = new_spot
                changed = True

            # Update instance name if we have fresh data
            if new_data["instance"] and provider_entry.get("instance") != new_data["instance"]:
                provider_entry["instance"] = new_data["instance"]
                changed = True

            if changed:
                updated_count += 1

    # Save only if something changed
    if updated_count > 0:
        with open(GPU_PRICING_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  gpu_pricing.json updated ({updated_count} prices changed)")
    else:
        print("  gpu_pricing.json unchanged (prices already up to date)")

    return updated_count > 0


if __name__ == "__main__":
    update_gpu_pricing()
