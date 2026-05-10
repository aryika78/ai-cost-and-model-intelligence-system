"""Tools for the Extraction Agent: document parsing and requirement saving."""

import json
import os
import tempfile
from langchain_core.tools import tool
from src.utils.document_parser import parse_document


@tool
def parse_uploaded_document(file_path: str) -> str:
    """Parse an uploaded document (PDF, DOCX, TXT, or image) and extract its text content.

    Args:
        file_path: Path to the uploaded file

    Returns:
        Extracted text from the document
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    file_ext = os.path.splitext(file_path)[1].lstrip(".")
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    text = parse_document(file_bytes, file_ext)

    if len(text) > 10000:
        text = text[:10000] + "\n\n[... truncated, document is very long ...]"

    return f"Extracted text from {os.path.basename(file_path)}:\n\n{text}"


@tool
def save_requirements(requirements: str) -> str:
    """Validate and save extracted requirements as structured JSON.

    The requirements should be a JSON string with these fields:
    - task_type: What the AI will do (e.g., "chatbot", "code generation", "document analysis")
    - use_case: Brief description of the use case
    - languages: List of languages needed (if applicable)
    - volume: Expected usage volume (requests/day, documents/day, etc.)
    - latency: Latency requirements ("real-time", "near-real-time", "batch")
    - accuracy_priority: How important accuracy is ("critical", "high", "moderate", "low")
    - budget: Budget constraints if any
    - deployment: Where it will run ("cloud_api", "self_hosted", "local", "any")
    - privacy: Privacy requirements ("strict", "moderate", "none")
    - additional_features: List of extra needs (e.g., ["vision", "function_calling", "streaming"])
    - specific_model: If user already has a specific model in mind
    - context_length_needs: Estimated context length needed
    - conversation_turns: Average conversation turns if applicable

    Args:
        requirements: JSON string of extracted requirements

    Returns:
        Validation result with what's complete and what's missing
    """
    try:
        reqs = json.loads(requirements)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON - {e}. Please provide valid JSON."

    critical_fields = ["task_type", "use_case"]
    important_fields = ["volume", "deployment", "latency"]
    optional_fields = [
        "languages", "accuracy_priority", "budget", "privacy",
        "additional_features", "specific_model", "context_length_needs",
        "conversation_turns",
    ]

    missing_critical = [f for f in critical_fields if not reqs.get(f)]
    missing_important = [f for f in important_fields if not reqs.get(f)]
    missing_optional = [f for f in optional_fields if not reqs.get(f)]

    present_fields = [f for f in reqs if reqs[f]]

    result = {
        "status": "incomplete" if missing_critical else "complete",
        "requirements": reqs,
        "present_fields": present_fields,
        "missing_critical": missing_critical,
        "missing_important": missing_important,
        "missing_optional": missing_optional,
    }

    if missing_critical:
        result["message"] = (
            f"Requirements are INCOMPLETE. Missing critical fields: {missing_critical}. "
            f"Also missing important fields: {missing_important}. "
            f"Please ask the user about these."
        )
    elif missing_important:
        result["message"] = (
            f"Requirements have all critical fields. Missing some important fields: {missing_important}. "
            f"Consider asking about these for better recommendations."
        )
    else:
        result["message"] = (
            f"Requirements are complete with {len(present_fields)} fields filled. "
            f"Ready for analysis."
        )

    return json.dumps(result, indent=2)
