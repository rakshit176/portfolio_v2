# tests/unit/test_providers.py
import pytest
from unittest.mock import Mock, patch
from utils.providers import get_llm, get_embeddings, ProviderConfig


@pytest.mark.asyncio
async def test_get_llm_returns_groq_by_default():
    """Should return Groq LLM when no provider specified."""
    with patch("utils.providers.ChatGroq") as mock_groq:
        mock_llm = Mock()
        mock_groq.return_value = mock_llm

        llm = get_llm()

        assert llm == mock_llm
        mock_groq.assert_called_once()


@pytest.mark.asyncio
async def test_get_llm_supports_provider_selection():
    """Should return correct LLM based on provider argument."""
    provider_classes = {
        "groq": "ChatGroq",
        "gemini": "ChatGoogleGenerativeAI",
        "openrouter": "ChatOpenRouter",
        "ollama": "ChatOllama",
    }

    for provider, class_name in provider_classes.items():
        with patch(f"utils.providers.{class_name}") as mock_class:
            mock_llm = Mock()
            mock_class.return_value = mock_llm

            llm = get_llm(provider=provider)

            assert llm == mock_llm


@pytest.mark.asyncio
async def test_get_embeddings_returns_ollama_embeddings():
    """Should return Ollama embeddings by default."""
    with patch("utils.providers.OllamaEmbeddings") as mock_ollama:
        mock_emb = Mock()
        mock_ollama.return_value = mock_emb

        emb = get_embeddings()

        assert emb == mock_emb
        mock_ollama.assert_called_once_with(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )


@pytest.mark.asyncio
async def test_provider_config_from_env():
    """Should load API keys from environment."""
    import os

    with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
        config = ProviderConfig.from_env()

        assert config.groq_api_key == "test_key"
