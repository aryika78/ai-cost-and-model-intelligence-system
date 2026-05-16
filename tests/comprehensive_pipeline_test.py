"""Comprehensive end-to-end pipeline tests covering all agent phases,
file upload, and multiple query types. Produces a quality analysis report.

Run: python tests/comprehensive_pipeline_test.py
"""

import json
import os
import sys
import time
import tempfile

# Force UTF-8 output on Windows (box-drawing chars, em-dashes, etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env so API keys are available when running directly
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed; keys must already be in environment


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_pdf(text: str) -> str:
    """Create a minimal text-based PDF file. Returns path to temp file."""
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        path = tempfile.mktemp(suffix=".pdf")
        c = rl_canvas.Canvas(path)
        y = 750
        for line in text.splitlines():
            c.drawString(50, y, line)
            y -= 14
            if y < 50:
                c.showPage()
                y = 750
        c.save()
        return path
    except ImportError:
        pass

    # Fallback: minimal valid PDF (text-based, no external lib needed)
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj

2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj

3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj

4 0 obj
<< /Length {content_len} >>
stream
BT /F1 12 Tf 50 750 Td
"""
    lines = text.replace("(", "\\(").replace(")", "\\)").splitlines()
    stream_lines = []
    y = 750
    for line in lines[:40]:
        stream_lines.append(f"({line}) Tj 0 -16 Td")
    stream_body = "\n".join(stream_lines) + "\nET"
    encoded = stream_body.encode()

    path = tempfile.mktemp(suffix=".pdf")
    with open(path, "wb") as f:
        header = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
            b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n\n"
        )
        content_block = (
            f"4 0 obj\n<< /Length {len(encoded)} >>\nstream\nBT /F1 12 Tf 50 750 Td\n"
            .encode() + encoded + b"\nendstream\nendobj\n\n"
        )
        font_block = (
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n\n"
            b"xref\n0 6\n0000000000 65535 f \n"
        )
        f.write(header + content_block + font_block)
    return path


def _make_txt(text: str) -> str:
    path = tempfile.mktemp(suffix=".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _quality_score(state: dict) -> dict:
    """Score the output quality of a completed pipeline run."""
    scores = {}
    phase = state.get("phase", "")
    scores["reached_complete"] = phase == "complete"
    scores["has_requirements"] = bool(state.get("requirements"))
    scores["has_recommended_models"] = bool(state.get("recommended_models"))
    scores["has_cost_report"] = bool(state.get("cost_report", "").strip())

    req = state.get("requirement_summary", {})
    scores["has_confidence_breakdown"] = bool(req)
    scores["stated_count"] = len(req.get("user_stated", []))
    scores["inferred_count"] = len(req.get("inferred", []))
    scores["assumed_count"] = len(req.get("assumed", []))

    cost_report = state.get("cost_report", "")
    scores["cost_has_range"] = "–" in cost_report or " - " in cost_report or "to $" in cost_report.lower()
    scores["cost_has_dollar"] = "$" in cost_report
    scores["cost_has_monthly"] = "month" in cost_report.lower()
    scores["cost_has_scenarios"] = any(w in cost_report.lower() for w in
                                        ["optimistic", "pessimistic", "realistic", "scenario"])
    scores["cost_has_assumptions"] = "assum" in cost_report.lower()

    # Count messages
    scores["message_count"] = len(state.get("messages", []))
    return scores


def _print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _print_result(label: str, value):
    ok = value if isinstance(value, bool) else bool(value)
    icon = "[OK]" if ok else "[--]"
    print(f"  {icon}  {label}: {value}")


# ─── Test Runners ─────────────────────────────────────────────────────────────

def run_scenario(name: str, user_input: str, uploaded_file: str = "",
                 follow_ups: list[str] | None = None) -> dict:
    """Run a single scenario through the full pipeline. Returns (state, duration)."""
    from src.agents.graph import run_pipeline

    print(f"\n  >> Input: {user_input[:120]}{'...' if len(user_input) > 120 else ''}")
    if uploaded_file:
        print(f"  >> File:  {os.path.basename(uploaded_file)}")

    t0 = time.time()
    state = run_pipeline(user_input, uploaded_file=uploaded_file)
    phase = state.get("phase", "?")

    # If stopped at extraction, provide follow-ups and re-run
    if follow_ups and phase in ("extraction", "wait_for_input"):
        for fu in follow_ups:
            print(f"\n  >> Follow-up: {fu[:100]}")
            state = run_pipeline(fu, current_state=state)
            phase = state.get("phase", "?")
            if phase == "complete":
                break

    elapsed = round(time.time() - t0, 1)
    print(f"  >> Phase reached: {phase} ({elapsed}s)")
    return state, elapsed


# ─── Individual Tests ─────────────────────────────────────────────────────────

def test_mode_a_gpt4o() -> dict:
    """Mode A: User specifies GPT-4o with usage details."""
    _print_section("TEST 1 — Mode A: GPT-4o with full usage details")
    state, elapsed = run_scenario(
        name="mode_a_gpt4o",
        user_input=(
            "I want to use openai/gpt-4o. "
            "We have 2,000 API requests per day. "
            "Average input is 600 tokens, output is 400 tokens. "
            "It's a real-time customer service chatbot, cloud API deployment. "
            "Conversations average 4 turns each."
        ),
    )
    scores = _quality_score(state)
    scores["elapsed"] = elapsed
    scores["scenario"] = "Mode A: GPT-4o"

    # Mode A specific checks
    reqs = state.get("requirements", {})
    scores["detected_mode_a"] = "specific_model" in reqs or any(
        "gpt-4o" in str(v).lower() for v in reqs.values()
    )

    _print_result("Reached complete phase", scores["reached_complete"])
    _print_result("Has requirements", scores["has_requirements"])
    _print_result("Detected Mode A (specific_model field)", scores["detected_mode_a"])
    _print_result("Has recommended models", scores["has_recommended_models"])
    if state.get("recommended_models"):
        print(f"       Models: {state['recommended_models']}")
    _print_result("Cost report present", scores["has_cost_report"])
    _print_result("Cost has $ amounts", scores["cost_has_dollar"])
    _print_result("Cost has range (low–high)", scores["cost_has_range"])
    _print_result("Cost has monthly figures", scores["cost_has_monthly"])
    _print_result("Cost has scenarios", scores["cost_has_scenarios"])

    if state.get("cost_report"):
        # Show first 600 chars of cost report
        print(f"\n  COST REPORT PREVIEW:\n{'-'*60}")
        print(state["cost_report"][:800])
        print(f"  ... [{len(state['cost_report'])} chars total]")

    return scores


def test_mode_b_chatbot() -> dict:
    """Mode B: Customer support chatbot — rich detail, no specific model."""
    _print_section("TEST 2 — Mode B: Customer support chatbot (full detail)")
    state, elapsed = run_scenario(
        name="mode_b_chatbot",
        user_input=(
            "We need an AI chatbot for customer support. We handle 5,000 queries per day. "
            "Each conversation averages 5 turns, with about 200 words per message. "
            "We need real-time responses under 2 seconds. "
            "Budget is $500 per month maximum. Cloud API deployment. English only. "
            "We need function calling to look up order status from our database."
        ),
    )
    scores = _quality_score(state)
    scores["elapsed"] = elapsed
    scores["scenario"] = "Mode B: Chatbot"

    reqs = state.get("requirements", {})
    scores["detected_function_calling"] = any(
        "function" in str(k).lower() or "function" in str(v).lower()
        for k, v in reqs.items()
    )
    scores["detected_budget"] = any(
        "budget" in str(k).lower() or "500" in str(v) for k, v in reqs.items()
    )

    _print_result("Reached complete phase", scores["reached_complete"])
    _print_result("Has requirements", scores["has_requirements"])
    _print_result("Detected function calling constraint", scores["detected_function_calling"])
    _print_result("Detected budget constraint", scores["detected_budget"])
    _print_result("Confidence breakdown present", scores["has_confidence_breakdown"])
    print(f"       Stated: {scores['stated_count']}  Inferred: {scores['inferred_count']}  Assumed: {scores['assumed_count']}")
    _print_result("Has recommended models", scores["has_recommended_models"])
    if state.get("recommended_models"):
        print(f"       Models: {state['recommended_models']}")
    _print_result("Cost has range", scores["cost_has_range"])
    _print_result("Cost has scenarios", scores["cost_has_scenarios"])
    _print_result("Cost has assumptions", scores["cost_has_assumptions"])

    if state.get("cost_report"):
        print(f"\n  COST REPORT PREVIEW:\n{'-'*60}")
        print(state["cost_report"][:600])
        print(f"  ... [{len(state['cost_report'])} chars total]")

    return scores


def test_mode_b_rag() -> dict:
    """Mode B: RAG document Q&A system."""
    _print_section("TEST 3 — Mode B: RAG document Q&A system")
    state, elapsed = run_scenario(
        name="mode_b_rag",
        user_input=(
            "We're building an internal document Q&A system. We have 1,000 PDF documents "
            "averaging 30 pages each (mostly technical manuals). "
            "Our team of 30 people will ask about 300 questions per day. "
            "We need accurate answers grounded in the documents — not hallucinations. "
            "Cloud deployment on AWS. Budget $300/month. "
            "The system should handle follow-up questions in the same context."
        ),
    )
    scores = _quality_score(state)
    scores["elapsed"] = elapsed
    scores["scenario"] = "Mode B: RAG Q&A"

    reqs = state.get("requirements", {})
    # RAG should detect embedding + retrieval as architecture
    scores["detected_rag_architecture"] = any(
        any(kw in str(v).lower() for kw in ["rag", "retriev", "embed", "vector"])
        for v in reqs.values()
    )

    _print_result("Reached complete phase", scores["reached_complete"])
    _print_result("Has requirements", scores["has_requirements"])
    _print_result("Detected RAG/retrieval architecture", scores["detected_rag_architecture"])
    _print_result("Has recommended models", scores["has_recommended_models"])
    if state.get("recommended_models"):
        print(f"       Models: {state['recommended_models']}")
    _print_result("Cost has range", scores["cost_has_range"])

    if state.get("cost_report"):
        print(f"\n  COST REPORT PREVIEW:\n{'-'*60}")
        print(state["cost_report"][:600])
        print(f"  ... [{len(state['cost_report'])} chars total]")

    return scores


def test_mode_b_vision() -> dict:
    """Mode B: Vision/image analysis."""
    _print_section("TEST 4 — Mode B: Vision — product image analysis")
    state, elapsed = run_scenario(
        name="mode_b_vision",
        user_input=(
            "I need to analyze product photos from our e-commerce platform. "
            "For each product image, I need to: extract product name, category, color, "
            "and condition from the photo. We process about 10,000 images per day in batch. "
            "24-hour delay is acceptable (it's a nightly pipeline). "
            "We need structured JSON output for each image. "
            "Cloud deployment. Budget: $200/month."
        ),
    )
    scores = _quality_score(state)
    scores["elapsed"] = elapsed
    scores["scenario"] = "Mode B: Vision"

    reqs = state.get("requirements", {})
    scores["detected_vision"] = any(
        any(kw in str(v).lower() for kw in ["vision", "image", "visual", "photo"])
        for v in reqs.values()
    )
    scores["detected_batch"] = any(
        any(kw in str(v).lower() for kw in ["batch", "nightly", "delay", "async"])
        for v in reqs.values()
    )
    scores["detected_structured_output"] = any(
        any(kw in str(v).lower() for kw in ["json", "struct", "output format"])
        for v in reqs.values()
    )

    _print_result("Reached complete phase", scores["reached_complete"])
    _print_result("Has requirements", scores["has_requirements"])
    _print_result("Detected vision modality", scores["detected_vision"])
    _print_result("Detected batch processing", scores["detected_batch"])
    _print_result("Detected structured output", scores["detected_structured_output"])
    _print_result("Has recommended models", scores["has_recommended_models"])
    if state.get("recommended_models"):
        print(f"       Models: {state['recommended_models']}")
    _print_result("Cost has range", scores["cost_has_range"])

    if state.get("cost_report"):
        print(f"\n  COST REPORT PREVIEW:\n{'-'*60}")
        print(state["cost_report"][:600])
        print(f"  ... [{len(state['cost_report'])} chars total]")

    return scores


def test_file_upload_txt() -> dict:
    """File upload: TXT requirements document."""
    _print_section("TEST 5 — File Upload: TXT requirements document")

    doc_text = """AI Project Requirements Document
Company: TechCorp India
Date: 2026-05-15

PROJECT: Code Review Automation Tool

OVERVIEW:
We want to automate code review for our 50 developer team.
The system should review pull requests and flag:
- Security vulnerabilities
- Code quality issues
- Performance bottlenecks

SCALE:
- ~200 PRs per day
- Average PR size: 500 lines of code (~3000 tokens)
- Need results within 5 minutes of PR submission

TECHNICAL REQUIREMENTS:
- Must support Python, JavaScript, TypeScript, Java
- Output must be structured (JSON) for our CI/CD integration
- Context window: at least 32K tokens (large PRs)
- Self-hosted preferred (code is proprietary, cannot send to external APIs)

BUDGET: $2000/month maximum for infrastructure

CURRENT SITUATION:
We currently use GPT-4 API but costs are unpredictable and code leaves our network.
"""

    txt_path = _make_txt(doc_text)
    print(f"  Created TXT file: {os.path.basename(txt_path)}")

    try:
        # First test: direct document parsing tool
        from src.tools.extraction_tools import parse_uploaded_document
        parse_result = parse_uploaded_document.invoke(txt_path)
        print(f"\n  parse_uploaded_document result ({len(parse_result)} chars):")
        print(f"  {parse_result[:200]}...")
        doc_parsed = "Extracted text" in parse_result and "TechCorp" in parse_result
        print(f"\n  [{'OK' if doc_parsed else '--'}] Document parsed correctly: {doc_parsed}")

        # Second test: full pipeline with file upload
        state, elapsed = run_scenario(
            name="file_upload_txt",
            user_input="Please analyze the attached requirements document and help me find the best AI solution.",
            uploaded_file=txt_path,
        )
        scores = _quality_score(state)
        scores["elapsed"] = elapsed
        scores["scenario"] = "File Upload: TXT"
        scores["doc_parsed_correctly"] = doc_parsed

        reqs = state.get("requirements", {})
        scores["detected_self_hosted"] = any(
            any(kw in str(v).lower() for kw in ["self-host", "self host", "on-prem", "private", "local"])
            for v in reqs.values()
        )
        scores["detected_large_context"] = any(
            any(kw in str(v).lower() for kw in ["32k", "context", "long"])
            for v in reqs.values()
        )

        _print_result("Document parsed correctly", doc_parsed)
        _print_result("Reached complete phase", scores["reached_complete"])
        _print_result("Detected self-hosted requirement", scores["detected_self_hosted"])
        _print_result("Detected large context requirement", scores["detected_large_context"])
        _print_result("Has recommended models", scores["has_recommended_models"])
        if state.get("recommended_models"):
            print(f"       Models: {state['recommended_models']}")
        _print_result("Has confidence breakdown", scores["has_confidence_breakdown"])
        _print_result("Cost has range", scores["cost_has_range"])

        if state.get("cost_report"):
            print(f"\n  COST REPORT PREVIEW:\n{'-'*60}")
            print(state["cost_report"][:600])
            print(f"  ... [{len(state['cost_report'])} chars total]")

        return scores

    finally:
        if os.path.exists(txt_path):
            os.unlink(txt_path)


def test_mode_b_ambiguous_with_followup() -> dict:
    """Mode B: Ambiguous query — tests follow-up Q&A flow."""
    _print_section("TEST 6 — Mode B: Ambiguous query with follow-up answers")

    state, elapsed = run_scenario(
        name="mode_b_ambiguous",
        user_input="I want to add AI to my app.",
        follow_ups=[
            "It's a mobile app for doctors. They need to describe patient symptoms by voice "
            "and get a differential diagnosis list. About 500 doctors, each using it 20 times/day. "
            "HIPAA compliance required. US market only. Budget around $1000/month.",
        ],
    )

    scores = _quality_score(state)
    scores["elapsed"] = elapsed
    scores["scenario"] = "Mode B: Ambiguous + Follow-up"

    reqs = state.get("requirements", {})
    scores["detected_hipaa"] = any(
        any(kw in str(v).lower() for kw in ["hipaa", "compliance", "healthcare", "medical", "privacy"])
        for v in reqs.values()
    )
    scores["detected_speech"] = any(
        any(kw in str(v).lower() for kw in ["speech", "voice", "audio", "transcri"])
        for v in reqs.values()
    )

    _print_result("Reached complete phase", scores["reached_complete"])
    _print_result("Detected HIPAA/compliance constraint", scores["detected_hipaa"])
    _print_result("Detected speech/voice modality", scores["detected_speech"])
    _print_result("Has recommended models", scores["has_recommended_models"])
    if state.get("recommended_models"):
        print(f"       Models: {state['recommended_models']}")
    _print_result("Cost has range", scores["cost_has_range"])
    _print_result("Cost has scenarios", scores["cost_has_scenarios"])

    if state.get("cost_report"):
        print(f"\n  COST REPORT PREVIEW:\n{'-'*60}")
        print(state["cost_report"][:600])
        print(f"  ... [{len(state['cost_report'])} chars total]")

    return scores


# ─── Quality Analysis Report ──────────────────────────────────────────────────

def print_quality_report(all_scores: list[dict]):
    """Print a final comprehensive quality analysis."""
    _print_section("FINAL QUALITY ANALYSIS REPORT")

    total = len(all_scores)
    completed = sum(1 for s in all_scores if s.get("reached_complete"))

    print(f"\n  PIPELINE COMPLETION")
    print(f"  {'─'*50}")
    print(f"  Scenarios run:       {total}")
    print(f"  Fully completed:     {completed}/{total}  ({100*completed//total}%)")
    print()

    print(f"  SCENARIO BREAKDOWN")
    print(f"  {'─'*50}")
    for s in all_scores:
        name = s.get("scenario", "?")
        phase = "[COMPLETE]" if s.get("reached_complete") else "[PARTIAL] "
        elapsed = s.get("elapsed", 0)
        models = s.get("has_recommended_models", False)
        cost = s.get("has_cost_report", False)
        print(f"  {phase}  {name:<35} {elapsed:>5}s  models={'Y' if models else 'N'}  cost={'Y' if cost else 'N'}")

    print()
    print(f"  EXTRACTION QUALITY")
    print(f"  {'─'*50}")
    for s in all_scores:
        if not s.get("has_requirements"):
            continue
        name = s.get("scenario", "?")
        stated = s.get("stated_count", 0)
        inferred = s.get("inferred_count", 0)
        assumed = s.get("assumed_count", 0)
        total_fields = stated + inferred + assumed
        print(f"  {name:<40} fields={total_fields:>2}  stated={stated}  inferred={inferred}  assumed={assumed}")

    print()
    print(f"  CONSTRAINT DETECTION")
    print(f"  {'─'*50}")
    constraint_checks = [
        ("Mode A: GPT-4o",              "detected_mode_a"),
        ("Mode B: Chatbot",             "detected_function_calling"),
        ("Mode B: Chatbot",             "detected_budget"),
        ("Mode B: RAG Q&A",             "detected_rag_architecture"),
        ("Mode B: Vision",              "detected_vision"),
        ("Mode B: Vision",              "detected_batch"),
        ("Mode B: Vision",              "detected_structured_output"),
        ("File Upload: TXT",            "detected_self_hosted"),
        ("File Upload: TXT",            "detected_large_context"),
        ("Mode B: Ambiguous + Follow-up", "detected_hipaa"),
        ("Mode B: Ambiguous + Follow-up", "detected_speech"),
    ]
    passed_constraints = 0
    for scenario_name, key in constraint_checks:
        for s in all_scores:
            if s.get("scenario") == scenario_name and key in s:
                ok = s[key]
                icon = "[OK]" if ok else "[--]"
                label = key.replace("detected_", "").replace("_", " ")
                print(f"  {icon}  {scenario_name:<35}   {label}")
                if ok:
                    passed_constraints += 1
                break
    print(f"\n  Constraints correctly detected: {passed_constraints}/{len(constraint_checks)}")

    print()
    print(f"  COST REPORT QUALITY")
    print(f"  {'─'*50}")
    cost_fields = ["cost_has_dollar", "cost_has_range", "cost_has_monthly",
                   "cost_has_scenarios", "cost_has_assumptions"]
    for s in all_scores:
        if not s.get("reached_complete"):
            continue
        name = s.get("scenario", "?")
        scores_str = "  ".join(
            f"{'Y' if s.get(f) else 'N'} {f.replace('cost_has_', '')}"
            for f in cost_fields
        )
        print(f"  {name:<40} {scores_str}")

    print()
    print(f"  OVERALL QUALITY VERDICT")
    print(f"  {'─'*50}")

    completion_rate = completed / total
    avg_elapsed = sum(s.get("elapsed", 0) for s in all_scores) / total

    if completion_rate >= 0.8:
        verdict = "EXCELLENT"
    elif completion_rate >= 0.6:
        verdict = "GOOD"
    elif completion_rate >= 0.4:
        verdict = "PARTIAL"
    else:
        verdict = "NEEDS WORK"

    print(f"  Pipeline Completion:    {100*completion_rate:.0f}%  — {verdict}")
    print(f"  Avg scenario time:      {avg_elapsed:.1f}s")
    print(f"  Constraint detection:   {passed_constraints}/{len(constraint_checks)} ({100*passed_constraints//len(constraint_checks)}%)")

    cost_complete = [s for s in all_scores if s.get("reached_complete")]
    if cost_complete:
        cost_quality = sum(
            sum(1 for f in cost_fields if s.get(f)) / len(cost_fields)
            for s in cost_complete
        ) / len(cost_complete)
        print(f"  Cost report quality:    {100*cost_quality:.0f}%  ({len(cost_complete)} reports)")

    print()
    print(f"  ISSUES AND OBSERVATIONS")
    print(f"  {'─'*50}")
    for s in all_scores:
        name = s.get("scenario", "?")
        issues = []
        if not s.get("reached_complete"):
            issues.append("Did not reach complete phase")
        if not s.get("has_recommended_models") and s.get("reached_complete"):
            issues.append("Reached complete but no recommended_models in state")
        if s.get("cost_has_dollar") is False and s.get("reached_complete"):
            issues.append("Cost report missing dollar amounts")
        if not issues:
            issues.append("No issues")
        print(f"  {name:<40}   {'; '.join(issues)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 70)
    print("  MCKH AI Engine — Comprehensive Pipeline Test Suite")
    print("  Testing: Extraction -> Analysis -> Cost (all phases)")
    print("  Scenarios: Mode A, Mode B ×3, File Upload, Ambiguous+Follow-up")
    print("=" * 70)

    from src.db.qdrant_manager import get_collection_count
    count = get_collection_count()
    print(f"\n  DB: {count} models loaded")
    if count == 0:
        print("  ERROR: DB is empty. Run `python -m src.updater.run_update` first.")
        sys.exit(1)

    print(f"  LLM_PROVIDER: {os.getenv('LLM_PROVIDER', 'groq')}")
    print()

    all_scores = []

    # Run all tests — catch failures individually so one bad test doesn't block others
    tests = [
        ("Mode A: GPT-4o", test_mode_a_gpt4o),
        ("Mode B: Chatbot", test_mode_b_chatbot),
        ("Mode B: RAG Q&A", test_mode_b_rag),
        ("Mode B: Vision", test_mode_b_vision),
        ("File Upload: TXT", test_file_upload_txt),
        ("Mode B: Ambiguous+Follow-up", test_mode_b_ambiguous_with_followup),
    ]

    for name, fn in tests:
        try:
            scores = fn()
            all_scores.append(scores)
        except Exception as e:
            import traceback
            print(f"\n  [EXCEPTION] {name}: {e}")
            traceback.print_exc()
            all_scores.append({
                "scenario": name,
                "reached_complete": False,
                "exception": str(e),
                "elapsed": 0,
            })

    print_quality_report(all_scores)
    print()

    # Save full report to file
    report_path = os.path.join(os.path.dirname(__file__), "pipeline_test_report.json")
    with open(report_path, "w") as f:
        json.dump(all_scores, f, indent=2, default=str)
    print(f"  Full report saved to: {report_path}")
    print()


if __name__ == "__main__":
    main()
