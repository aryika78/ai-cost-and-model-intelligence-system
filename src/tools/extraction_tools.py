"""Tools for the Extraction Agent: document parsing and requirement saving."""

import json
import os
from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool
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


class _RequirementsSchema(BaseModel):
    """Schema for save_requirements.

    requirements: dict of {field_name: {value, confidence, source, impacts}}
    extraction_complete: True when ready to hand off to analysis
    reasoning: why extraction is complete or what's still missing
    rerun_analysis: True if model selection needs to be re-evaluated (only relevant after a completed pipeline)
    """
    requirements: dict = Field(
        default_factory=dict,
        description=(
            "All extracted requirement fields. Each key is a descriptive field name. "
            "Each value is either a plain value or a dict with: "
            "value (the extracted value), "
            "confidence (user_stated | inferred | assumed), "
            "source (where it came from), "
            "impacts (what it affects downstream)."
        )
    )
    extraction_complete: bool = Field(default=False)
    reasoning: str = Field(default="")
    rerun_analysis: bool = Field(
        default=True,
        description=(
            "Whether model selection needs to be re-evaluated. "
            "Set to False if only cost parameters changed and existing model recommendations are still valid. "
            "Set to True if the model choice itself might change (new requirements, different model requested, etc.). "
            "Only relevant when following up after a completed pipeline — ignored on first run."
        )
    )


def _save_requirements_fn(requirements: dict, extraction_complete: bool = False, reasoning: str = "", rerun_analysis: bool = True) -> str:
    """Process requirements dict into a structured result."""
    flat_reqs = {}
    stated, inferred, assumed = [], [], []

    for key, val in requirements.items():
        if isinstance(val, dict) and "value" in val:
            flat_reqs[key] = val["value"]
            conf = val.get("confidence", "assumed")
            if conf == "user_stated":
                stated.append(key)
            elif conf == "inferred":
                inferred.append(key)
            else:
                assumed.append(key)
        else:
            flat_reqs[key] = val
            assumed.append(key)

    result = {
        "status": "complete" if extraction_complete else "incomplete",
        "extraction_complete": extraction_complete,
        "rerun_analysis": rerun_analysis,
        "requirements": flat_reqs,
        "requirement_summary": {
            "user_stated": stated,
            "inferred": inferred,
            "assumed": assumed,
        },
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
            f"{len(inferred)} inferred, {len(assumed)} assumed. Ready for analysis."
        )
    else:
        result["message"] = (
            f"Requirements incomplete ({len(stated)} stated so far). "
            f"Reason: {reasoning or 'More information needed.'}"
        )

    return json.dumps(result, indent=2)


save_requirements = StructuredTool.from_function(
    func=_save_requirements_fn,
    name="save_requirements",
    description=(
        "Save extracted requirements and signal completion status. "
        "Pass ALL requirements together as the 'requirements' dict, where each key is a descriptive "
        "field name and each value is a dict with: value, confidence (user_stated|inferred|assumed), "
        "source, impacts. "
        "Set extraction_complete=True when ready to proceed to model analysis. "
        "Set extraction_complete=False when saving partial progress and asking follow-up questions. "
        "Set rerun_analysis=False when following up after a completed pipeline and only cost parameters changed — "
        "the existing model recommendations are still valid. Set rerun_analysis=True when model selection needs re-evaluation."
    ),
    args_schema=_RequirementsSchema,
)
