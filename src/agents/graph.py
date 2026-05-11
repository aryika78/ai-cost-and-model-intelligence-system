"""LangGraph StateGraph orchestrating Extraction → Analysis → Cost agents."""

import json
import os
import re
import time
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.agents.extraction_agent import create_extraction_agent
from src.agents.analysis_agent import create_analysis_agent
from src.agents.cost_agent import create_cost_agent


def _reset_agent_cache():
    """Clear all cached agents so they are recreated with the current LLM_PROVIDER."""
    global _extraction_agent, _analysis_agent, _cost_agent
    _extraction_agent = None
    _analysis_agent = None
    _cost_agent = None


def _invoke_with_fallback(create_fn, messages: list, retries: int = 2) -> dict:
    """Invoke an agent. On Groq rate limit: wait then retry; if still failing,
    switch to Cerebras automatically (if CEREBRAS_API_KEY is set).
    """
    agent = create_fn()
    for attempt in range(retries + 1):
        try:
            return agent.invoke({"messages": messages})
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower() or "rate limit" in err.lower()
            if not is_rate_limit:
                raise

            # Last attempt — try Cerebras if available
            if attempt == retries - 1 and os.getenv("CEREBRAS_API_KEY"):
                print("  [FALLBACK] Groq rate-limited, switching to Cerebras...")
                os.environ["LLM_PROVIDER"] = "cerebras"
                _reset_agent_cache()  # Force fresh agent with new provider
                agent = create_fn()
                continue

            if attempt < retries:
                wait = 30 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s before retry {attempt + 1}/{retries}...")
                time.sleep(wait)
            else:
                raise


class AgentState(TypedDict):
    """Shared state across all agents."""
    messages: list  # Full conversation history
    requirements: dict  # Extracted requirements (flat key-value)
    requirement_summary: dict  # Confidence breakdown: {user_stated, inferred, assumed}
    recommended_models: list  # Model IDs from analysis
    cost_report: str  # Final cost report
    phase: str  # "extraction", "analysis", "cost", "complete"
    user_input: str  # Latest user input
    uploaded_file: str  # Path to uploaded file, if any


# Create agents (lazy initialization)
_extraction_agent = None
_analysis_agent = None
_cost_agent = None


def _get_extraction_agent():
    global _extraction_agent
    if _extraction_agent is None:
        _extraction_agent = create_extraction_agent()
    return _extraction_agent


def _get_analysis_agent():
    global _analysis_agent
    if _analysis_agent is None:
        _analysis_agent = create_analysis_agent()
    return _analysis_agent


def _get_cost_agent():
    global _cost_agent
    if _cost_agent is None:
        _cost_agent = create_cost_agent()
    return _cost_agent


def extraction_node(state: AgentState) -> dict:
    """Run the extraction agent to gather/refine requirements."""
    agent = _get_extraction_agent()

    # Build messages for extraction agent
    messages = list(state.get("messages", []))
    user_input = state.get("user_input", "")

    if user_input:
        messages.append(HumanMessage(content=user_input))

    # If there's an uploaded file, mention it
    uploaded = state.get("uploaded_file", "")
    if uploaded:
        messages.append(HumanMessage(
            content=f"The user uploaded a file at: {uploaded}. Please parse it to extract requirements."
        ))

    result = _invoke_with_fallback(_get_extraction_agent, messages)
    agent_messages = result.get("messages", [])

    # Get the last AI message as the response
    response = ""
    for msg in reversed(agent_messages):
        if isinstance(msg, AIMessage) and msg.content:
            response = msg.content
            break

    # Check if requirements were saved (look for save_requirements call results)
    requirements = state.get("requirements", {})
    requirement_summary = state.get("requirement_summary", {})
    is_complete = False

    for msg in agent_messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            if '"extraction_complete"' in msg.content:
                try:
                    parsed = json.loads(msg.content)
                    if parsed.get("requirements"):
                        requirements = parsed["requirements"]
                    if parsed.get("requirement_summary"):
                        requirement_summary = parsed["requirement_summary"]
                    if parsed.get("extraction_complete"):
                        is_complete = True
                except (json.JSONDecodeError, KeyError):
                    pass

    new_messages = state.get("messages", [])
    if user_input:
        new_messages = new_messages + [HumanMessage(content=user_input)]
    new_messages = new_messages + [AIMessage(content=response)]

    return {
        "messages": new_messages,
        "requirements": requirements,
        "requirement_summary": requirement_summary,
        "phase": "analysis" if is_complete else "extraction",
    }


def analysis_node(state: AgentState) -> dict:
    """Run the analysis agent to find and recommend models."""
    agent = _get_analysis_agent()

    requirements = state.get("requirements", {})
    req_str = json.dumps(requirements, indent=2)

    messages = [
        HumanMessage(content=f"Here are the user's requirements:\n\n{req_str}\n\n"
                     f"Please find and recommend the best AI models for these requirements.")
    ]

    # If a specific model was requested (Mode A)
    specific_model = requirements.get("specific_model")
    if specific_model:
        messages = [
            HumanMessage(content=f"The user specifically wants to use model: {specific_model}\n\n"
                         f"Full requirements:\n{req_str}\n\n"
                         f"Please look up this model and provide details. "
                         f"Also suggest 1-2 alternatives if relevant.")
        ]

    result = _invoke_with_fallback(_get_analysis_agent, messages)
    agent_messages = result.get("messages", [])

    # Get the response
    response = ""
    for msg in reversed(agent_messages):
        if isinstance(msg, AIMessage) and msg.content:
            response = msg.content
            break

    # Extract recommended model IDs from response
    recommended = []
    # Look for "RECOMMENDED MODELS: [...]" pattern
    match = re.search(r'RECOMMENDED MODELS:\s*\[([^\]]+)\]', response)
    if match:
        ids_str = match.group(1)
        recommended = [mid.strip().strip('"\'') for mid in ids_str.split(",")]
    elif specific_model:
        recommended = [specific_model]

    new_messages = state.get("messages", []) + [AIMessage(content=response)]

    return {
        "messages": new_messages,
        "recommended_models": recommended,
        "phase": "cost",
    }


def cost_node(state: AgentState) -> dict:
    """Run the cost agent to calculate costs for recommended models."""
    agent = _get_cost_agent()

    requirements = state.get("requirements", {})
    recommended = state.get("recommended_models", [])
    req_str = json.dumps(requirements, indent=2)

    prompt = (
        f"User requirements:\n{req_str}\n\n"
        f"Recommended models: {recommended}\n\n"
        f"Please calculate comprehensive cost estimates for each model. "
        f"Remember to provide cost RANGES (optimistic/realistic/pessimistic), not fixed numbers. "
        f"Show all assumptions explicitly."
    )

    result = _invoke_with_fallback(_get_cost_agent, [HumanMessage(content=prompt)])
    agent_messages = result.get("messages", [])

    response = ""
    for msg in reversed(agent_messages):
        if isinstance(msg, AIMessage) and msg.content:
            response = msg.content
            break

    new_messages = state.get("messages", []) + [AIMessage(content=response)]

    return {
        "messages": new_messages,
        "cost_report": response,
        "phase": "complete",
    }


def route_after_extraction(state: AgentState) -> str:
    """Route: if extraction complete, go to analysis; otherwise loop back."""
    if state.get("phase") == "analysis":
        return "analysis"
    return "wait_for_input"


def route_after_analysis(state: AgentState) -> str:
    """Route: after analysis, always go to cost."""
    return "cost"


def wait_for_input_node(state: AgentState) -> dict:
    """Placeholder node that signals we need more user input."""
    return {"phase": "extraction"}


def build_graph() -> StateGraph:
    """Build the full LangGraph pipeline."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("extraction", extraction_node)
    graph.add_node("wait_for_input", wait_for_input_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("cost", cost_node)

    # Set entry point
    graph.set_entry_point("extraction")

    # Add edges
    graph.add_conditional_edges(
        "extraction",
        route_after_extraction,
        {"analysis": "analysis", "wait_for_input": "wait_for_input"},
    )
    graph.add_edge("wait_for_input", END)  # Returns to user for more input
    graph.add_edge("analysis", "cost")
    graph.add_edge("cost", END)

    return graph.compile()


# Module-level compiled graph
pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = build_graph()
    return pipeline


def run_pipeline(user_input: str, current_state: dict | None = None,
                 uploaded_file: str = "") -> dict:
    """Run the pipeline with user input, continuing from current state.

    Args:
        user_input: The user's message
        current_state: Previous state to continue from (None for new conversation)
        uploaded_file: Path to uploaded file if any

    Returns:
        Updated state dict
    """
    graph = get_pipeline()

    if current_state is None:
        state = AgentState(
            messages=[],
            requirements={},
            requirement_summary={},
            recommended_models=[],
            cost_report="",
            phase="extraction",
            user_input=user_input,
            uploaded_file=uploaded_file,
        )
    else:
        state = dict(current_state)
        state["user_input"] = user_input
        state["uploaded_file"] = uploaded_file
        # If we were waiting, re-enter extraction
        if state.get("phase") in ("extraction", "wait_for_input"):
            state["phase"] = "extraction"

    result = graph.invoke(state)
    return result
