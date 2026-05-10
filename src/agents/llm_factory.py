"""LLM factory for local Ollama or Groq-backed agents."""

import os

from langchain_groq import ChatGroq


def create_chat_model(temperature: float = 0.2):
    """Create the chat model used by agents.

    Defaults to Ollama for local, no-cost development. Set LLM_PROVIDER=groq
    to use the previous Groq-hosted model.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            temperature=temperature,
        )

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=temperature,
    )
