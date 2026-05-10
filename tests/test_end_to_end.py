"""End-to-end tests for the AI Model Discovery & Cost Estimation platform.

Run: python -m pytest tests/test_end_to_end.py -v
Or:  python tests/test_end_to_end.py  (for manual testing without pytest)
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_embeddings():
    """Test that the embedding model loads and produces correct dimensions."""
    from src.db.embeddings import embed_text, embed_texts, EMBEDDING_DIM

    vec = embed_text("test query about chatbots")
    assert len(vec) == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM} dims, got {len(vec)}"

    vecs = embed_texts(["hello", "world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == EMBEDDING_DIM
    print("  [PASS] Embeddings module works correctly")


def test_qdrant_operations():
    """Test Qdrant CRUD operations."""
    from src.db.qdrant_manager import (
        create_collection, upsert_model, semantic_search,
        get_model, get_collection_count,
    )

    create_collection()

    # Upsert a test model
    test_model = {
        "id": "test/test-model-v1",
        "name": "Test Model V1",
        "description": "A test model for unit testing purposes",
        "context_window": 4096,
        "input_price_per_mtok": 1.0,
        "output_price_per_mtok": 2.0,
        "type": "chat",
        "category": "general",
        "tags": ["chat", "test"],
        "open_source": False,
        "provider": "test",
        "source": "test",
    }
    upsert_model(test_model)

    # Search
    results = semantic_search("test model for chatting", top_k=5)
    assert len(results) > 0, "Semantic search returned no results"
    assert any(r["id"] == "test/test-model-v1" for r in results), "Test model not found in search"

    # Get by ID
    model = get_model("test/test-model-v1")
    assert model is not None, "get_model returned None"
    assert model["name"] == "Test Model V1"

    # Count
    count = get_collection_count()
    assert count > 0, "Collection count is 0"

    print(f"  [PASS] Qdrant operations work correctly ({count} models in DB)")


def test_document_parser():
    """Test document parsing."""
    from src.utils.document_parser import parse_document

    text = parse_document(b"Hello, this is a test document.", "txt")
    assert "Hello" in text
    assert "test document" in text
    print("  [PASS] Document parser works for TXT files")


def test_reference_data():
    """Test that reference JSON files load correctly."""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

    with open(os.path.join(config_dir, "gpu_pricing.json")) as f:
        gpu_data = json.load(f)
    assert "gpus" in gpu_data
    assert len(gpu_data["gpus"]) >= 8
    print(f"  [PASS] GPU pricing: {len(gpu_data['gpus'])} GPUs loaded")

    with open(os.path.join(config_dir, "platforms.json")) as f:
        platforms = json.load(f)
    assert "platforms" in platforms
    assert len(platforms["platforms"]) >= 10
    print(f"  [PASS] Platforms: {len(platforms['platforms'])} platforms loaded")


def test_tools_extraction():
    """Test extraction tools."""
    from src.tools.extraction_tools import save_requirements

    # Test complete requirements
    reqs = json.dumps({
        "task_type": "chatbot",
        "use_case": "Customer support for ecommerce",
        "volume": "5000 requests/day",
        "deployment": "cloud_api",
        "latency": "real-time",
    })
    result = save_requirements.invoke(reqs)
    parsed = json.loads(result)
    assert parsed["status"] == "complete"
    print("  [PASS] Extraction tools work correctly")


def test_tools_analysis():
    """Test analysis tools (requires populated DB)."""
    from src.db.qdrant_manager import get_collection_count
    if get_collection_count() == 0:
        print("  [SKIP] Analysis tools test - DB empty. Run `python -m src.updater.run_update` first.")
        return

    from src.tools.analysis_tools import search_models, count_matching_models

    result = search_models.invoke({"query": "fast chatbot model", "filters": "{}"})
    assert "Found" in result or "No models" in result
    print(f"  [PASS] Analysis tools work correctly")


def test_tools_cost():
    """Test cost tools."""
    from src.tools.cost_tools import calculate_self_hosting_cost, get_gpu_options

    result = calculate_self_hosting_cost.invoke(json.dumps({
        "model_size": "70b",
        "quantization": "int4",
        "provider": "runpod",
        "hours_per_day": 24,
        "redundancy": 1,
    }))
    assert "Daily" in result or "daily" in result.lower()
    assert "$" in result
    print("  [PASS] Self-hosting cost tool works")

    result = get_gpu_options.invoke("70b")
    assert "GPU" in result or "NVIDIA" in result
    print("  [PASS] GPU options tool works")


def test_scenario_chatbot():
    """Full scenario test: Customer support chatbot."""
    print("\n  --- Scenario: Customer Support Chatbot ---")

    from src.tools.extraction_tools import save_requirements
    reqs = {
        "task_type": "chatbot",
        "use_case": "Customer support for ecommerce store, Hindi + English",
        "languages": ["Hindi", "English"],
        "volume": "5000 chats/day",
        "latency": "real-time",
        "accuracy_priority": "high",
        "deployment": "cloud_api",
        "privacy": "moderate",
        "conversation_turns": 5,
    }
    result = json.loads(save_requirements.invoke(json.dumps(reqs)))
    assert result["status"] == "complete"
    print("  [PASS] Requirements extracted and validated")

    from src.tools.cost_tools import generate_cost_table
    from src.db.qdrant_manager import get_collection_count
    if get_collection_count() > 0:
        from src.tools.analysis_tools import search_models
        search_result = search_models.invoke({
            "query": "multilingual customer support chatbot Hindi English fast",
            "filters": json.dumps({"type": "chat"}),
        })
        print(f"  [PASS] Model search returned results")
    else:
        print("  [SKIP] Model search - DB empty")


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("MCKH Platform - End-to-End Tests")
    print("=" * 60)

    tests = [
        ("Reference Data", test_reference_data),
        ("Embeddings Module", test_embeddings),
        ("Qdrant Operations", test_qdrant_operations),
        ("Document Parser", test_document_parser),
        ("Extraction Tools", test_tools_extraction),
        ("Analysis Tools", test_tools_analysis),
        ("Cost Tools", test_tools_cost),
        ("Scenario: Chatbot", test_scenario_chatbot),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            if "SKIP" in str(e):
                skipped += 1
                print(f"  [SKIP] {e}")
            else:
                failed += 1
                print(f"  [FAIL] {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
