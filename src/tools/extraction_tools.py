"""Tools for the Extraction Agent: document parsing and requirement saving."""

import json
import os
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
    """Save extracted requirements with confidence tagging.

    The requirements should be a JSON object where each extracted field has:
    {
      "field_name": {
        "value": "the actual value",
        "confidence": "user_stated" | "inferred" | "assumed",
        "source": "brief explanation of why"
      },
      ...,
      "extraction_complete": true | false,
      "reasoning": "why extraction is complete or what's still missing"
    }

    Confidence levels:
    - user_stated: user explicitly said this
    - inferred: reasonably deduced from context
    - assumed: typical default, not mentioned by user

    Use ANY field names that make sense for this request — do not use a fixed template.

    Args:
        requirements: JSON string of confidence-tagged requirements

    Returns:
        Validation result with summary of stated/inferred/assumed fields
    """
    try:
        data = json.loads(requirements)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON - {e}. Please provide valid JSON."

    extraction_complete = data.pop("extraction_complete", False)
    reasoning = data.pop("reasoning", "")

    # Parse each field — support confidence-tagged or plain values
    flat_reqs = {}
    stated, inferred, assumed = [], [], []

    for key, val in data.items():
        if isinstance(val, dict) and "value" in val and "confidence" in val:
            flat_reqs[key] = val["value"]
            conf = val.get("confidence", "assumed")
            if conf == "user_stated":
                stated.append(key)
            elif conf == "inferred":
                inferred.append(key)
            else:
                assumed.append(key)
        else:
            # Plain value — treat as assumed (no explicit confidence given)
            flat_reqs[key] = val
            assumed.append(key)

    requirement_summary = {
        "user_stated": stated,
        "inferred": inferred,
        "assumed": assumed,
    }

    result = {
        "status": "complete" if extraction_complete else "incomplete",
        "extraction_complete": extraction_complete,
        "requirements": flat_reqs,
        "requirement_summary": requirement_summary,
        "counts": {
            "stated": len(stated),
            "inferred": len(inferred),
            "assumed": len(assumed),
        },
    }

    if reasoning:
        result["reasoning"] = reasoning

    if extraction_complete:
        result["message"] = (
            f"Requirements captured: {len(stated)} user-stated, "
            f"{len(inferred)} inferred, {len(assumed)} assumed. "
            f"Ready for analysis."
        )
    else:
        result["message"] = (
            f"Requirements incomplete ({len(stated)} stated so far). "
            f"Reason: {reasoning or 'More information needed.'}"
        )

    return json.dumps(result, indent=2)
