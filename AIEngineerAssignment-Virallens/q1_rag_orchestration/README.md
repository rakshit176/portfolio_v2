# Q1 RAG Orchestration System

Multi-agent RAG (Retrieval-Augmented Generation) system with LangGraph orchestration.

## Architecture

The system consists of 4 agents orchestrated via LangGraph:

1. **Router Agent** - Classifies queries and determines routing
2. **Retriever Agent** - Fetches relevant context from Qdrant vector store
3. **Reasoning Agent** - Generates answers using retrieved context
4. **Critic Agent** - Evaluates answer faithfulness and completeness

### Flow

```
User Query → Router → [Retriever → Reasoning → Critic] → Final Answer
                     ↑___________________| (retry loop)
```

## Features

- **Multi-Provider Support**: Groq, Gemini, OpenRouter, Ollama with automatic fallback
- **Rate Limiting**: Token bucket with exponential backoff (30 RPM default)
- **Semantic Caching**: Redis-based caching with SHA256 cache keys
- **Vector Search**: Qdrant with MMR retrieval and cross-encoder reranking
- **Observability**: Structured JSON logging with trace IDs

## Setup

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- API keys (see `.env.example`)

### Installation

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install langchain-groq langchain-google-genai langchain-ollama langchain-openrouter

# Copy environment file
cp .env.example .env
# Edit .env with your API keys
```

### Start Services

```bash
docker-compose up -d
```

This starts:
- Qdrant (port 6333/6334)
- Redis (port 6379)

### Ingest Documents

```bash
python main.py ingest ./data
```

### Run Queries

```bash
python main.py
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `QDRANT_URL` | Qdrant URL (default: http://localhost:6334) |
| `REDIS_URL` | Redis URL (default: redis://localhost:6379/0) |
| `OLLAMA_BASE_URL` | Ollama URL (default: http://localhost:11434) |

## Testing

```bash
# Run all tests
python -m pytest tests/unit/ -v

# Run specific test file
python -m pytest tests/unit/test_router_agent.py -v

# With coverage
python -m pytest tests/unit/ --cov=. --cov-report=html
```

## Sample Queries

The system includes 5 sample queries:

1. "What are the main revenue drivers?"
2. "Explain the company's product strategy."
3. "Who are the key executives mentioned in the documents?"
4. "What risks are outlined in the annual report?"
5. "Describe the financial performance for Q4."

## Project Structure

```
q1_rag_orchestration/
├── agents/           # Agent implementations
│   ├── router_agent.py
│   ├── retriever_agent.py
│   ├── reasoning_agent.py
│   └── critic_agent.py
├── graph/            # LangGraph workflow
│   ├── state.py      # RAGState TypedDict
│   └── workflow.py   # Graph definition
├── rag/              # RAG utilities
│   ├── chunker.py    # Text chunking
│   └── ingestor.py   # Document ingestion
├── utils/            # Utilities
│   ├── cache.py      # Redis caching
│   ├── logger.py     # Structured logging
│   ├── providers.py  # LLM provider factory
│   ├── prompts.py    # Prompt templates
│   └── rate_limiter.py
├── tests/            # Unit tests
├── docker-compose.yml
├── main.py           # Entry point
└── requirements.txt
```

## License

MIT
