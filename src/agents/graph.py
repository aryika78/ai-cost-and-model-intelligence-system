"""LangGraph StateGraph orchestrating Extraction → Analysis → Cost agents."""

import json
import os
import time
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
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


# Ordered fallback chain — tried in sequence when a provider rate-limits or errors.
# Each entry: (provider_name, required_env_var)
_FALLBACK_CHAIN = [
    ("mistral",      "MISTRAL_API_KEY"),
    ("openrouter",   "OPENROUTER_API_KEY"),
    ("groq",         "GROQ_API_KEY"),
    ("cerebras",     "CEREBRAS_API_KEY"),
    ("gemini",       "GEMINI_API_KEY"),
]


def _next_available_provider(current: str) -> str | None:
    """Return the next provider in the fallback chain that has a key set."""
    found_current = False
    for name, env_var in _FALLBACK_CHAIN:
        if name == current:
            found_current = True
            continue
        if found_current and os.getenv(env_var):
            return name
    return None


def _invoke_with_fallback(create_fn, messages: list, retries: int = 2) -> dict:
    """Invoke an agent with automatic provider fallback on rate limits.

    On rate limit / queue overflow: waits 15s then retries once, then
    automatically switches to the next available provider in _FALLBACK_CHAIN.
    Cycles through all available providers before giving up.
    """
    tried_providers = set()

    while True:
        current_provider = os.getenv("LLM_PROVIDER", "mistral").lower()
        tried_providers.add(current_provider)
        agent = create_fn()

        for attempt in range(retries + 1):
            try:
                return agent.invoke({"messages": messages})
            except Exception as e:
                err = str(e)
                is_rate_limit = (
                    "429" in err
                    or "rate_limit" in err.lower()
                    or "rate limit" in err.lower()
                    or "queue" in err.lower()
                    or "too_many_requests" in err.lower()
                )
                is_connection = (
                    "connection" in err.lower()
                    or "getaddrinfo" in err.lower()
                    or "connecterror" in err.lower()
                )

                if not is_rate_limit and not is_connection:
                    raise  # Non-rate-limit error — don't retry with different provider

                if attempt < retries:
                    wait = 15 * (attempt + 1)
                    print(f"  [RATE LIMIT] {current_provider} — waiting {wait}s (retry {attempt + 1}/{retries})...")
                    time.sleep(wait)
                else:
                    # Exhausted retries on this provider — try next
                    next_provider = _next_available_provider(current_provider)

                    # Skip already-tried providers
                    while next_provider and next_provider in tried_providers:
                        os.environ["LLM_PROVIDER"] = next_provider
                        next_provider = _next_available_provider(next_provider)

                    if next_provider:
                        print(f"  [FALLBACK] {current_provider} exhausted → switching to {next_provider}...")
                        os.environ["LLM_PROVIDER"] = next_provider
                        _reset_agent_cache()
                        break  # Break retry loop, outer while will re-invoke with new provider
                    else:
                        raise RuntimeError(
                            f"All providers exhausted ({', '.join(tried_providers)}). "
                            f"Last error: {err}"
                        )
        else:
            continue  # retries loop finished normally — shouldn't happen
        # Broke out of retry loop (switching provider) — continue outer while
        continue


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
    rerun_analysis: bool  # Whether model selection needs re-evaluation on follow-up


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
    rerun_analysis = True  # Default: re-run analysis on follow-ups

    for msg in agent_messages:
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "save_requirements":
            try:
                parsed = json.loads(msg.content)
                if parsed.get("requirements"):
                    requirements = parsed["requirements"]
                if parsed.get("requirement_summary"):
                    requirement_summary = parsed["requirement_summary"]
                if parsed.get("extraction_complete"):
                    is_complete = True
                if "rerun_analysis" in parsed:
                    rerun_analysis = parsed["rerun_analysis"]
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
        "rerun_analysis": rerun_analysis,
        "phase": "analysis" if is_complete else "extraction",
    }


def analysis_node(state: AgentState) -> dict:
    """Run the analysis agent to find and recommend models."""
    agent = _get_analysis_agent()

    requirements = state.get("requirements", {})
    req_str = json.dumps(requirements, indent=2)

    # Include full conversation history so agent understands context and any follow-up changes
    history = list(state.get("messages", []))
    history.append(HumanMessage(content=f"Here are the extracted requirements:\n\n{req_str}\n\n"
                     f"Please analyze these and recommend the best AI models."))

    result = _invoke_with_fallback(_get_analysis_agent, history)
    agent_messages = result.get("messages", [])

    # Get the response (last AI text message)
    response = ""
    for msg in reversed(agent_messages):
        if isinstance(msg, AIMessage) and msg.content:
            response = msg.content
            break

    # Extract recommended model IDs from save_recommendations tool call result
    recommended = []
    for msg in agent_messages:
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "save_recommendations":
            try:
                parsed = json.loads(msg.content)
                if parsed.get("recommended_models"):
                    recommended = parsed["recommended_models"]
                    break
            except (json.JSONDecodeError, KeyError):
                pass

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

    # Include full conversation history so agent understands context and any follow-up changes
    history = list(state.get("messages", []))
    history.append(HumanMessage(content=prompt))

    result = _invoke_with_fallback(_get_cost_agent, history)
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
    """Route: if extraction complete, go to analysis or cost depending on rerun_analysis flag."""
    if state.get("phase") == "analysis":
        if state.get("rerun_analysis", True):
            return "analysis"
        else:
            return "cost"  # Skip analysis — existing recommendations still valid
    return "wait_for_input"


def route_after_analysis(state: AgentState) -> str:
    """Route: go to cost only if analysis identified specific models.
    If recommended_models is empty, stop here — the analysis response already
    explains why (no matching models, conflicting constraints, etc.).
    Running cost with an empty list causes hallucination.
    """
    if state.get("recommended_models"):
        return "cost"
    return "no_models_found"


def wait_for_input_node(state: AgentState) -> dict:
    """Placeholder node that signals we need more user input."""
    return {"phase": "extraction"}


def no_models_found_node(state: AgentState) -> dict:
    """Analysis completed but no specific models were identified.
    The analysis agent's written response already explains why.
    Mark as complete so the UI shows that response as the final output.
    """
    return {"phase": "complete", "cost_report": ""}


def build_graph() -> StateGraph:
    """Build the full LangGraph pipeline."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("extraction", extraction_node)
    graph.add_node("wait_for_input", wait_for_input_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("no_models_found", no_models_found_node)
    graph.add_node("cost", cost_node)

    # Set entry point
    graph.set_entry_point("extraction")

    # Add edges
    graph.add_conditional_edges(
        "extraction",
        route_after_extraction,
        {"analysis": "analysis", "cost": "cost", "wait_for_input": "wait_for_input"},
    )
    graph.add_edge("wait_for_input", END)  # Returns to user for more input
    graph.add_conditional_edges(
        "analysis",
        route_after_analysis,
        {"cost": "cost", "no_models_found": "no_models_found"},
    )
    graph.add_edge("no_models_found", END)
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
        # Re-enter extraction for any follow-up (waiting, or post-completion)
        if state.get("phase") in ("extraction", "wait_for_input", "complete"):
            state["phase"] = "extraction"

    result = graph.invoke(state)
    return result
