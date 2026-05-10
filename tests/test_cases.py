"""Test cases for model search accuracy and cost estimation correctness."""

from src.tools.analysis_tools import search_models, get_model_details, compare_models
from src.tools.cost_tools import calculate_api_cost, calculate_self_hosting_cost, calculate_scenario_costs

def test_case_1_multilingual_chatbot():
    print("=" * 60)
    print("TEST 1: Hindi-English multilingual chatbot, 5000 chats/day")
    print("=" * 60)
    result = search_models.invoke({"query": "Hindi English multilingual chatbot conversational AI", "filters": "{}"})
    print(result[:2500])


def test_case_2_code_generation():
    print("=" * 60)
    print("TEST 2: Code generation model, open-source preferred")
    print("=" * 60)
    result = search_models.invoke({
        "query": "code generation Python JavaScript programming assistant",
        "filters": '{"type": "chat", "open_source": true}'
    })
    print(result[:2500])


def test_case_3_api_cost_gpt4o():
    print("=" * 60)
    print("TEST 3: GPT-4o cost for 5000 requests/day")
    print("=" * 60)
    import json
    params = json.dumps({
        "model_id": "openai/gpt-4o",
        "avg_input_tokens": 500,
        "avg_output_tokens": 200,
        "requests_per_day": 5000,
        "cache_hit_rate": 0.1,
        "batch_percentage": 0.0,
        "conversation_turns": 3,
        "agent_calls_per_request": 1
    })
    result = calculate_api_cost.invoke({"params": params})
    print(result)


def test_case_4_self_hosting_cost_llama70b():
    print("=" * 60)
    print("TEST 4: Self-hosting cost for Llama 70B on RunPod")
    print("=" * 60)
    import json
    params = json.dumps({
        "model_size": "70b",
        "quantization": "int4",
        "provider": "runpod",
        "hours_per_day": 24,
        "redundancy": 1
    })
    result = calculate_self_hosting_cost.invoke({"params": params})
    print(result)


def test_case_5_scenario_costs():
    print("=" * 60)
    print("TEST 5: Scenario costs for gpt-4o-mini")
    print("=" * 60)
    import json
    params = json.dumps({
        "model_id": "openai/gpt-4o-mini",
        "scenarios": {
            "optimistic": {
                "requests_per_day": 3500,
                "avg_input_tokens": 300,
                "avg_output_tokens": 150,
                "conversation_turns": 2,
                "agent_calls_per_request": 1,
                "cache_hit_rate": 0.3,
                "batch_percentage": 0.0
            },
            "realistic": {
                "requests_per_day": 5000,
                "avg_input_tokens": 500,
                "avg_output_tokens": 200,
                "conversation_turns": 3,
                "agent_calls_per_request": 1,
                "cache_hit_rate": 0.15,
                "batch_percentage": 0.0
            },
            "pessimistic": {
                "requests_per_day": 8000,
                "avg_input_tokens": 800,
                "avg_output_tokens": 400,
                "conversation_turns": 5,
                "agent_calls_per_request": 2,
                "cache_hit_rate": 0.05,
                "batch_percentage": 0.0
            }
        }
    })
    result = calculate_scenario_costs.invoke({"params": params})
    print(result)


def test_case_6_rag_document_analysis():
    print("=" * 60)
    print("TEST 6: RAG / document analysis, long context, self-hosted")
    print("=" * 60)
    result = search_models.invoke({
        "query": "long context document analysis RAG summarization legal",
        "filters": '{"open_source": true}'
    })
    print(result[:2500])


def test_case_7_image_model():
    print("=" * 60)
    print("TEST 7: Image generation model (Stable Diffusion alternatives)")
    print("=" * 60)
    result = search_models.invoke({
        "query": "image generation text-to-image diffusion model",
        "filters": '{"type": "image_generation"}'
    })
    print(result[:2500])


if __name__ == "__main__":
    test_case_1_multilingual_chatbot()
    print()
    test_case_2_code_generation()
    print()
    test_case_3_api_cost_gpt4o()
    print()
    test_case_4_self_hosting_cost_llama70b()
    print()
    test_case_5_scenario_costs()
    print()
    test_case_6_rag_document_analysis()
    print()
    test_case_7_image_model()
