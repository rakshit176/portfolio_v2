# utils/providers.py
import os
from typing import Optional
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openrouter import ChatOpenRouter


class ProviderConfig(BaseModel):
    """Configuration for all LLM providers."""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    qdrant_url: str = "http://localhost:6334"
    redis_url: str = "redis://localhost:6379/0"
    ollama_base_url: str = "http://localhost:11434"

    @classmethod
    def from_env(cls) -> "ProviderConfig":
        """Load configuration from environment variables."""
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6334"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )


def get_llm(
    provider: str = "groq",
    model: Optional[str] = None,
    config: Optional[ProviderConfig] = None
):
    """
    Get LLM instance for specified provider.

    Args:
        provider: Provider name (groq, gemini, openrouter, ollama)
        model: Model name (uses provider default if None)
        config: Provider config (uses env vars if None)

    Returns:
        LangChain Chat LLM instance
    """
    if config is None:
        config = ProviderConfig.from_env()

    provider_models = {
        "groq": ("llama3-70b-8192", ChatGroq, {"api_key": config.groq_api_key}),
        "gemini": ("gemini-1.5-flash", ChatGoogleGenerativeAI, {"api_key": config.gemini_api_key}),
        "openrouter": ("meta-llama/llama-3-70b", ChatOpenRouter, {"openrouter_api_key": config.openrouter_api_key}),
        "ollama": ("llama3", ChatOllama, {"base_url": config.ollama_base_url}),
    }

    if provider not in provider_models:
        raise ValueError(f"Unknown provider: {provider}")

    default_model, llm_class, kwargs = provider_models[provider]
    model_name = model or default_model

    return llm_class(model=model_name, **kwargs)


def get_embeddings(
    model: str = "nomic-embed-text",
    config: Optional[ProviderConfig] = None
):
    """
    Get embeddings instance.

    Args:
        model: Model name (default: nomic-embed-text via Ollama)
        config: Provider config

    Returns:
        LangChain embeddings instance
    """
    if config is None:
        config = ProviderConfig.from_env()

    return OllamaEmbeddings(
        model=model,
        base_url=config.ollama_base_url
    )
