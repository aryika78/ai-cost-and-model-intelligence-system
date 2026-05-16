"""LLM factory: Mistral → Groq → Cerebras → Together → Gemini → Ollama.

Free tier limits:
- Mistral:  500,000 TPM / 1B tokens/month (mistral-small-latest)  ← PRIMARY, best limits
- Together: $25 free credit, Llama 3.3 70B @ $0.88/M tok          ← backup #1
- Groq:     6,000 TPM  (llama-3.3-70b-versatile)                  ← backup #2
- Cerebras: 60,000 TPM (qwen-3-235b-a22b-instruct-2507)           ← backup #3
- Gemini:   gemini-2.5-flash via OpenAI-compat endpoint            ← backup #4

Set LLM_PROVIDER=mistral|together|groq|cerebras|gemini|ollama in .env.
"""

import os


def create_chat_model(temperature: float = 0.2):
    """Create the chat model used by agents."""
    provider = os.getenv("LLM_PROVIDER", "mistral").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            temperature=temperature,
        )

    if provider == "cerebras":
        return _cerebras_model(temperature)

    if provider == "gemini":
        return _gemini_model(temperature)

    if provider == "together":
        return _together_model(temperature)

    if provider == "mistral":
        return _mistral_model(temperature)

    if provider == "openrouter":
        return _openrouter_model(temperature)

    # Default: Groq
    return _groq_model(temperature)


def _groq_model(temperature: float):
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=temperature,
    )


def _cerebras_model(temperature: float):
    """Cerebras via OpenAI-compatible client (langchain-openai)."""
    from langchain_openai import ChatOpenAI
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY not set")
    # qwen-3-235b is best but often queued; llama3.1-8b is the reliable fallback
    model = os.getenv("CEREBRAS_MODEL", "qwen-3-235b-a22b-instruct-2507")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1",
    )


def _gemini_model(temperature: float):
    """Gemini 2.5-flash via OpenAI-compatible endpoint (langchain-openai)."""
    from langchain_openai import ChatOpenAI
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_tokens=4000,  # thinking model needs explicit token budget
    )


def _mistral_model(temperature: float):
    """Mistral Small via OpenAI-compatible endpoint. Free tier: 500K TPM, 1B tok/month."""
    from langchain_openai import ChatOpenAI
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set")
    model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
    )


def _together_model(temperature: float):
    """Together AI via OpenAI-compatible endpoint. $25 free credit, Llama 3.3 70B."""
    from langchain_openai import ChatOpenAI
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY not set")
    model = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://api.together.xyz/v1",
    )


def _openrouter_model(temperature: float):
    """OpenRouter free models — Llama 3.3 70B :free, Qwen3 72B :free, etc.
    All :free models are completely free, no charges. 200 req/day per model.
    """
    from langchain_openai import ChatOpenAI
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    # qwen3-72b:free has less upstream contention than llama-3.3-70b:free
    # Override via OPENROUTER_MODEL env var if needed
    model = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-72b:free")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "https://github.com/mckh-ai", "X-Title": "MCKH AI Engine"},
    )
