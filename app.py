"""MCKH Technologies - AI Model Discovery & Cost Estimation Platform.

Streamlit-based web interface for the AI model recommendation and cost estimation system.
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

load_dotenv()

st.set_page_config(
    page_title="MCKH AI Cost & Model Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar ---
with st.sidebar:
    st.title("MCKH Technologies")
    st.caption("AI Model Discovery & Cost Estimation")
    st.divider()

    # Database status
    st.subheader("Database Status")
    try:
        from src.db.qdrant_manager import get_collection_count
        count = get_collection_count()
        if count > 0:
            st.success(f"{count} models in database")
        else:
            st.warning("Database empty")
            if st.button("Populate Database", type="primary"):
                with st.spinner("Fetching models from OpenRouter, LiteLLM, HuggingFace..."):
                    from src.updater.run_update import run_update
                    run_update()
                    st.rerun()
    except Exception as e:
        st.error(f"DB error: {e}")
        if st.button("Initialize Database"):
            from src.updater.run_update import run_update
            with st.spinner("Initializing..."):
                run_update()
                st.rerun()

    st.divider()

    # File upload
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Upload requirements (PDF, DOCX, TXT, Image)",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
        help="Upload a project brief or requirements document",
    )

    st.divider()

    if st.button("Start New Analysis", type="secondary", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.divider()
    st.markdown("""
    **How it works:**
    1. Describe what you need (or upload a document)
    2. We extract your requirements through conversation
    3. We search 1000+ models to find the best fit
    4. We calculate cost ranges (not fixed numbers!)

    **Two modes:**
    - **Mode A:** You specify a model → we calculate costs
    - **Mode B:** You describe your needs → we find models + costs
    """)


# --- Session State Init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = None
if "phase" not in st.session_state:
    st.session_state.phase = "extraction"
if "processing" not in st.session_state:
    st.session_state.processing = False


# --- Main Content ---
st.title("AI Model Discovery & Cost Estimation")

# Phase indicator
phase = st.session_state.phase
phase_labels = {
    "extraction": "Step 1: Understanding Your Requirements",
    "analysis": "Step 2: Finding Best Models",
    "cost": "Step 3: Calculating Costs",
    "complete": "Analysis Complete",
}
phase_icons = {
    "extraction": "💬",
    "analysis": "🔍",
    "cost": "💰",
    "complete": "✅",
}

col1, col2, col3, col4 = st.columns(4)
for i, (p, label) in enumerate(phase_labels.items()):
    col = [col1, col2, col3, col4][i]
    with col:
        if p == phase:
            st.markdown(f"**{phase_icons[p]} {label}**")
        elif list(phase_labels.keys()).index(p) < list(phase_labels.keys()).index(phase):
            st.markdown(f"~~{phase_icons[p]} {label}~~")
        else:
            st.markdown(f"*{phase_icons[p]} {label}*")

st.divider()

# --- Chat Display ---
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Welcome! I'm the MCKH AI Cost & Model Intelligence Engine.\n\n"
            "I can help you:\n"
            "- **Find the right AI model** for your project\n"
            "- **Estimate costs** with realistic ranges (not just fixed numbers)\n\n"
            "Tell me about your project, or try one of these:\n"
            "- *\"I want to build a customer support chatbot for 5000 chats/day\"*\n"
            "- *\"What would GPT-4o cost for 10,000 requests per day?\"*\n"
            "- *\"I need a code generation model, self-hosted, privacy critical\"*\n\n"
            "You can also upload a requirements document in the sidebar."
        )
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- Handle File Upload ---
uploaded_file_path = ""
if uploaded_file is not None and "file_processed" not in st.session_state:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        uploaded_file_path = tmp.name

    st.session_state.file_processed = True
    st.session_state.messages.append({
        "role": "user",
        "content": f"📄 Uploaded: {uploaded_file.name}"
    })


# --- Chat Input ---
user_input = st.chat_input(
    "Describe your AI project or ask about a specific model...",
    disabled=st.session_state.processing,
)

if user_input or uploaded_file_path:
    input_text = user_input or f"I uploaded a requirements document: {uploaded_file.name}. Please analyze it."

    # Show user message
    if user_input:
        st.session_state.messages.append({"role": "user", "content": input_text})
        with st.chat_message("user"):
            st.markdown(input_text)

    # Check API key for configured provider
    _provider = os.getenv("LLM_PROVIDER", "groq").lower()
    _missing_key = (
        (_provider == "groq" and not os.getenv("GROQ_API_KEY")) or
        (_provider == "cerebras" and not os.getenv("CEREBRAS_API_KEY"))
    )
    if _missing_key:
        _key_url = "https://console.groq.com/keys" if _provider == "groq" else "https://cloud.cerebras.ai"
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Please set your {_provider.upper()}_API_KEY in .env to get started. "
                       f"Get a free key at {_key_url}"
        })
        st.rerun()

    # Check database
    try:
        from src.db.qdrant_manager import get_collection_count
        if get_collection_count() == 0:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "The model database is empty. Please click 'Populate Database' in the sidebar first."
            })
            st.rerun()
    except Exception:
        pass

    # Process with agents
    st.session_state.processing = True

    with st.chat_message("assistant"):
        current_phase = st.session_state.phase

        if current_phase in ("extraction", "wait_for_input"):
            phase_msg = "Understanding your requirements..."
        elif current_phase == "analysis":
            phase_msg = "Searching models..."
        elif current_phase == "cost":
            phase_msg = "Calculating costs..."
        else:
            phase_msg = "Processing..."

        with st.spinner(phase_msg):
            try:
                from src.agents.graph import run_pipeline

                result = run_pipeline(
                    user_input=input_text,
                    current_state=st.session_state.agent_state,
                    uploaded_file=uploaded_file_path,
                )

                # Update state
                st.session_state.agent_state = result
                new_phase = result.get("phase", "extraction")
                st.session_state.phase = new_phase

                # Get the latest AI response
                messages = result.get("messages", [])
                response = ""
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        response = msg.content
                        break

                if response:
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                    })

                    # Show confidence breakdown when extraction completes
                    if new_phase == "analysis" and result.get("requirement_summary"):
                        summary = result["requirement_summary"]
                        stated = summary.get("user_stated", [])
                        inferred = summary.get("inferred", [])
                        assumed = summary.get("assumed", [])
                        with st.expander(
                            f"Requirements breakdown: {len(stated)} stated, "
                            f"{len(inferred)} inferred, {len(assumed)} assumed"
                        ):
                            if stated:
                                st.markdown(f"**You told us:** {', '.join(stated)}")
                            if inferred:
                                st.markdown(f"**We inferred:** {', '.join(inferred)}")
                            if assumed:
                                st.markdown(
                                    f"**We assumed (typical defaults):** {', '.join(assumed)}  \n"
                                    f"*Correct us if any assumption is wrong.*"
                                )

                    # Show recommended models when available
                    if new_phase in ("cost", "complete") and result.get("recommended_models"):
                        models = result["recommended_models"]
                        st.info(f"Recommended models: {', '.join(models)}")

                    # If complete, show summary
                    if new_phase == "complete" and result.get("cost_report"):
                        st.success("Analysis complete! See cost breakdown above.")

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I encountered an error: {error_msg}\n\nPlease try again."
                })

    st.session_state.processing = False
    st.rerun()
