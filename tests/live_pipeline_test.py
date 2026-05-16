"""Live pipeline test — runs real agent calls and shows full output per agent."""

import sys
import io
import os
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.agents.graph import run_pipeline

DIVIDER = "=" * 80
THIN = "-" * 60


def print_section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def print_agent_trace(result: dict, scenario_name: str):
    """Print a detailed trace of what each agent did."""
    messages = result.get("messages", [])

    print(f"\n{'='*80}")
    print(f"  SCENARIO: {scenario_name}")
    print(f"{'='*80}")

    # ---- EXTRACTION PHASE ----
    print(f"\n{'--- EXTRACTION AGENT OUTPUT ' + '-'*40}")
    # Find the last AI message before analysis starts
    # (the extraction agent's final response to the user)
    extraction_response = ""
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content and len(msg.content) > 50:
            extraction_response = msg.content
            break  # first substantive AI message = extraction agent response

    if extraction_response:
        print(extraction_response[:1500])
        if len(extraction_response) > 1500:
            print(f"  ... [truncated, {len(extraction_response)} chars total]")

    # Requirements extracted
    reqs = result.get("requirements", {})
    summary = result.get("requirement_summary", {})
    print(f"\n  REQUIREMENTS EXTRACTED ({len(reqs)} fields):")
    for k, v in reqs.items():
        print(f"    {k}: {v}")
    if summary:
        print(f"\n  CONFIDENCE BREAKDOWN:")
        print(f"    User stated:  {summary.get('user_stated', [])}")
        print(f"    Inferred:     {summary.get('inferred', [])}")
        print(f"    Assumed:      {summary.get('assumed', [])}")

    # ---- ANALYSIS PHASE ----
    print(f"\n{'--- ANALYSIS AGENT OUTPUT ' + '-'*41}")
    # Find analysis agent messages — look for tool calls (search_models, etc.)
    tool_calls_made = []
    analysis_response = ""

    in_analysis = False
    for msg in messages:
        # Analysis starts after extraction's AI message
        if isinstance(msg, HumanMessage) and "requirements" in msg.content.lower():
            in_analysis = True
        if not in_analysis:
            continue

        # Tool messages = search_models, get_model_details, save_recommendations calls
        if isinstance(msg, ToolMessage):
            # Trim tool output for display
            content_preview = msg.content[:300] if msg.content else ""
            tool_calls_made.append(f"    [Tool result preview]: {content_preview}...")

        if isinstance(msg, AIMessage) and msg.content and in_analysis:
            analysis_response = msg.content

    # Show tool calls
    if tool_calls_made:
        print(f"  Tools called by analysis agent ({len(tool_calls_made)} calls):")
        for tc in tool_calls_made[:6]:  # cap at 6
            print(tc[:200])

    # Show recommended models
    recommended = result.get("recommended_models", [])
    print(f"\n  RECOMMENDED MODELS: {recommended}")

    # Show analysis narrative
    if analysis_response:
        print(f"\n  ANALYSIS AGENT NARRATIVE (last response):")
        print(analysis_response[:2000])
        if len(analysis_response) > 2000:
            print(f"  ... [truncated, {len(analysis_response)} chars total]")

    # ---- COST PHASE ----
    print(f"\n{'--- COST AGENT OUTPUT ' + '-'*45}")
    cost_report = result.get("cost_report", "")
    if cost_report:
        print(cost_report[:3000])
        if len(cost_report) > 3000:
            print(f"  ... [truncated, {len(cost_report)} chars total]")
    else:
        print("  (No cost report — pipeline stopped before cost node)")

    # ---- FINAL STATUS ----
    print(f"\n  FINAL PHASE: {result.get('phase', '?')}")
    print(f"  Recommended models count: {len(recommended)}")
    print(f"  Cost report present: {'YES' if cost_report else 'NO'}")


def run_scenario(name: str, query: str, uploaded_file: str = ""):
    print_section(f"RUNNING: {name}")
    print(f"  Query: {query[:200]}")
    if uploaded_file:
        print(f"  File:  {uploaded_file}")
    print()

    try:
        # Step 1: Send user query (extraction phase)
        state = run_pipeline(user_input=query, uploaded_file=uploaded_file)

        # If extraction needs follow-up, send a generic "proceed" message
        phase = state.get("phase", "")
        if phase == "extraction":
            print("  [Extraction asked follow-up — sending 'proceed with best assumptions']")
            state = run_pipeline(
                user_input="Please proceed with your best assumptions for anything unclear.",
                current_state=state,
            )

        print_agent_trace(state, name)

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()


# ─── SCENARIOS ───────────────────────────────────────────────────────────────

# 1. MODE A: User specifies a model
run_scenario(
    name="MODE A — GPT-4o cost for customer support",
    query="I want to use GPT-4o for a customer support chatbot. We handle 10,000 conversations per day, average conversation is 8 messages, each message about 100 words.",
)

# 2. MODE B — Specific, clear requirements
run_scenario(
    name="MODE B — RAG pipeline, privacy-sensitive",
    query="I need to build a RAG system for a law firm. All data is confidential — must be self-hosted or on-premise. About 500 queries per day, documents are legal contracts up to 100 pages each.",
)

# 3. MODE B — Vague requirements (tests how well extraction clarifies)
run_scenario(
    name="MODE B — Vague startup query",
    query="I'm building an AI startup and need a good model. Budget is limited.",
)

# 4. MODE B — Multi-modal requirements
run_scenario(
    name="MODE B — Vision + code generation",
    query="Building a tool that takes screenshots of websites and generates code to replicate them. Need to process about 200 images per day, generate HTML/CSS code from each.",
)

# 5. MODE B — BRS PDF document upload
run_scenario(
    name="MODE B — BRS.pdf document upload",
    query="I've uploaded our Business Requirements Specification document. Please analyze it and recommend the best AI models with cost estimates.",
    uploaded_file=r"C:\MYWORK\At AI_ML_Int\At Internal Task\docs\BRS.pdf",
)
