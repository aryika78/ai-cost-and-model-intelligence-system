"""LLM factory: Groq → Cerebras fallback → Ollama.

Free tier limits:
- Groq:     6,000 TPM  (llama-3.3-70b-versatile)   ← primary, best for tool use
- Cerebras: 60,000 TPM (qwen-3-235b / llama3.1-8b) ← auto-fallback on rate limit

Set LLM_PROVIDER=groq or LLM_PROVIDER=cerebras in .env.
"""

import os


def create_chat_model(temperature: float = 0.2):
    """Create the chat model used by agents."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            temperature=temperature,
        )

    if provider == "cerebras":
        return _cerebras_model(temperature)

    # Default: Groq with Cerebras fallback
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
    model = os.getenv("CEREBRAS_MODEL", "qwen-3-235b")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1",
    )
