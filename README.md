# AI Cost & Model Intelligence Engine

> Find the right AI model for your project. Get real cost ranges — not guesses.

---

## What It Does

Companies adopting AI face two problems:
1. **Which model should I use?** — 500+ models exist, picking the right one is hard
2. **What will it cost?** — Budget overruns kill AI projects

This system solves both. You describe your project (or upload a requirements doc), and it:
- Searches 500+ AI models semantically to find the best fit
- Calculates cost **ranges** (optimistic / realistic / pessimistic) using real pricing data
- Compares API hosting vs self-hosting on GPU clouds

**Two modes:**
- **Mode A:** You pick a model → system calculates cost ranges
- **Mode B:** You describe your needs → system finds models + calculates costs

---

## Why Better Than Asking ChatGPT

| | ChatGPT | This System |
|---|---|---|
| Pricing | Guesses from memory | Real data from OpenRouter + LiteLLM APIs |
| Cost output | Single number | Optimistic / Realistic / Pessimistic ranges |
| Model coverage | ~20-30 known models | 500+ models with semantic search |
| Self-hosting | Vague estimates | Real GPU prices (RunPod, Lambda, AWS, GCP) |
| Reproducibility | Different every time | Same data, same math, same answer |

---

## Architecture

```
User Input (text or PDF)
        │
        ▼
┌─────────────────┐
│ Extraction Agent│  ← Groq Llama 3.3 70B
│ (requirements)  │    Asks follow-up questions
└────────┬────────┘    Handles PDF/DOCX/TXT upload
         │
         ▼
┌─────────────────┐
│ Analysis Agent  │  ← Semantic search on Qdrant
│ (model search)  │    500+ models, meaning-based
└────────┬────────┘    Multi-angle search + validation
         │
         ▼
┌─────────────────┐
│  Cost Agent     │  ← Real pricing data
│ (cost ranges)   │    7 specialized calculators
└─────────────────┘    API + self-hosting + fine-tuning
```

**Data pipeline (run once, refresh anytime):**
```
OpenRouter API ──┐
HuggingFace API ─┼──► Deduplicate ──► LLM Enrichment ──► Qdrant Vector DB
LiteLLM JSON ────┘                    (capability profiles)
```

---

## Setup

### 1. Clone & Install
```bash
git clone <repo-url>
cd Mckh
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
```
Edit `.env`:
```
GROQ_API_KEY=your_key_here     # Get free at console.groq.com/keys
LLM_PROVIDER=groq              # Use groq for enrichment (recommended)
```

### 3. Populate Database
```bash
python -m src.updater.run_update
```
Fetches 500+ models from OpenRouter, HuggingFace, LiteLLM. Enriches each with an LLM-generated capability profile via Groq (~35 min for 500 models at free-tier rate limits). Only needs to run once — re-runs skip already-enriched models.

### 4. Run the App
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## Testing

```bash
# Unit tests (fast, no API needed)
python tests/test_end_to_end.py

# Full accuracy test (needs populated DB + Groq API key)
python tests/comprehensive_test.py
```

The comprehensive test scores:
- **Search relevance** — 10 real-world queries (legal, code, medical, image gen, etc.)
- **Cost accuracy** — 4 pricing scenarios vs known reference prices
- **Range quality** — verifies optimistic/pessimistic spread is meaningful (≥2x)

---

## Project Structure

```
Mckh/
├── app.py                          # Streamlit UI
├── config/
│   ├── gpu_pricing.json            # GPU specs + prices (RunPod, Lambda, AWS, GCP, Azure)
│   └── platforms.json              # AI platform metadata
├── src/
│   ├── agents/
│   │   ├── graph.py                # LangGraph pipeline orchestration
│   │   ├── extraction_agent.py     # Requirement extraction agent
│   │   ├── analysis_agent.py       # Model search + recommendation agent
│   │   └── cost_agent.py           # Cost calculation agent
│   ├── tools/
│   │   ├── extraction_tools.py     # save_requirements, parse_uploaded_document
│   │   ├── analysis_tools.py       # search_models, compare_models, get_model_details
│   │   └── cost_tools.py           # 7 cost calculators
│   ├── db/
│   │   ├── qdrant_manager.py       # Vector DB operations (search, upsert, filter)
│   │   └── embeddings.py           # Local sentence-transformer embeddings
│   ├── updater/
│   │   ├── run_update.py           # Pipeline orchestrator
│   │   ├── openrouter_sync.py      # Fetch from OpenRouter API
│   │   ├── huggingface_sync.py     # Fetch from HuggingFace API
│   │   ├── litellm_sync.py         # Fetch pricing from LiteLLM
│   │   └── capability_enricher.py  # LLM-generated capability profiles
│   └── utils/
│       └── document_parser.py      # PDF, DOCX, TXT, image parsing
├── tests/
│   ├── test_end_to_end.py          # Unit tests
│   └── comprehensive_test.py       # Accuracy + relevance tests
└── data/                           # Qdrant local DB (auto-generated, gitignored)
```

---

## Cost Calculators

| Tool | What It Calculates |
|---|---|
| `calculate_api_cost` | API inference cost with caching, batching, multi-turn context |
| `calculate_scenario_costs` | Optimistic / Realistic / Pessimistic ranges |
| `generate_cost_table` | Cost at 100 / 500 / 1K / 5K / 10K / 50K requests/day |
| `calculate_self_hosting_cost` | GPU rental: maps model size → VRAM → GPU → provider price |
| `get_gpu_options` | All GPUs that can run a model, with pricing across providers |
| `calculate_embedding_cost` | RAG pipeline: initial embed + re-embed + query costs |
| `calculate_finetuning_cost` | Fine-tuning cost by platform and training tokens |

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Vector DB | Qdrant (local) | Semantic search, runs as a file, free |
| Embeddings | all-MiniLM-L6-v2 | Local, no API cost, 384 dims |
| LLM (agents) | Llama 3.3 70B via Groq | Free tier, fast, good quality |
| Agent framework | LangGraph | Multi-agent pipeline with state + routing |
| Frontend | Streamlit | Python-only, fast to build, free hosting |
| Data sources | OpenRouter + HuggingFace + LiteLLM | Free public APIs, real pricing |

---