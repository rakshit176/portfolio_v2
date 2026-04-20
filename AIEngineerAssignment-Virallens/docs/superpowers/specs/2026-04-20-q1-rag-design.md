# Q1 RAG System Design — Multi-Agent Orchestration

**Date:** 2026-04-20
**Status:** Approved
**Author:** Rakshit Kumar

---

## 1. Problem Statement

Build a production-grade multi-agent Retrieval-Augmented Generation (RAG) workflow using LangChain, LangGraph, and DeepAgents that:

1. Processes PDF documents from the provided data directory
2. Answers natural language queries with grounded, cited responses
3. Handles rate limiting across multiple free API providers (Groq, Gemini, OpenRouter)
4. Degrades gracefully when APIs are unavailable (fallback to Ollama)
5. Operates reliably in a Dockerized environment with persistent vector storage

### Success Criteria

- **Faithfulness**: All claims are grounded in retrieved source chunks
- **Reliability**: System remains responsive even with API failures
- **Performance**: Sub-5-second response time for typical queries
- **Reproducibility**: One-command setup via Docker Compose
- **Observability**: Full traceability via structured logging

---

## 2. Architecture Overview

### 2.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                              │
└────────────────────────────────┬────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Router Agent (llama3-8b)                                       │
│  ✓ Classify: factual | conversational | ambiguous               │
│  ✓ Decompose complex queries                                   │
│  ✓ Check semantic cache first                                  │
└────────────────────────────────┬────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Retriever Agent (nomic-embed-text + CrossEncoder)              │
│  ✓ Embed queries with batch processing                         │
│  ✓ Qdrant similarity search with MMR                           │
│  ✓ Cross-encoder reranking (top-10)                            │
│  ✓ Response caching of retrieved chunks                        │
└────────────────────────────────┬────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Reasoning Agent (llama3-70b → gemini-1.5-flash fallback)      │
│  ✓ Construct context-augmented prompt                          │
│  ✓ Generate grounded answer with citations                     │
│  ✓ Self-assess confidence (low/medium/high)                    │
│  ✓ Emit reasoning trace for Critic                             │
└────────────────────────────────┬────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Critic Agent (llama3-8b)                                       │
│  ✓ Faithfulness: Check claims vs citations                     │
│  ✓ Completeness: Address all sub-queries?                     │
│  ✓ Coherence: Logical structure                                │
│  ✓ Verdict: approve | retry (max 2) | escalate                │
└────────────────────────────────┬────────────────────────────────┘
                                 ▼
                    ┌────────────┴────────────┐
                    ▼                         ▼
              [Approve]                  [Retry]
                    │                         │
                    ▼                    ┌───┘
            Final Answer              Reasoning Agent
                  └──(with critique)───┘
```

### 2.2 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Orchestration** | LangGraph StateGraph | Agent coordination and state management |
| **Runtime** | DeepAgents | Lifecycle management, tool registry, streaming |
| **LLM Providers** | Groq, Gemini, OpenRouter, Ollama | Multi-provider with fallback chain |
| **Vector DB** | Qdrant v1.12.0 | Production-grade similarity search |
| **Embeddings** | nomic-embed-text (Ollama) | Local, free, 768-dim vectors |
| **Reranker** | cross-encoder-ms-marco-MiniLM-L-6-v2 | Local precision ranking |
| **Cache** | Redis (semantic) + TTL (response) | Deduplication and speed |
| **Logging** | Structured JSON with trace IDs | Observability and debugging |
| **Containerization** | Docker Compose | Reproducible deployment |

---

## 3. Components

### 3.1 Router Agent

**File:** `agents/router_agent.py`

**Responsibilities:**
- Parse and normalize raw user queries
- Classify query type: `factual`, `conversational`, or `ambiguous`
- Decompose complex multi-part questions into atomic sub-queries
- Check semantic cache before proceeding
- Attach routing metadata to graph state

**Model:** Groq `llama3-8b-8192` (fast, free tier, low latency)

**Inputs:**
```python
{
    "query": str,               # Raw user question
    "chat_history": list[dict]  # Prior conversation turns
}
```

**Outputs:**
```python
{
    "query_type": str,          # "factual" | "conversational" | "ambiguous"
    "sub_queries": list[str],   # Decomposed sub-questions
    "route": str,               # "retriever" | "reasoner" | "clarify"
    "cache_key": str | None     # Semantic cache key if hit
}
```

### 3.2 Retriever Agent

**File:** `agents/retriever_agent.py`

**Responsibilities:**
- Convert queries to embedding vectors via nomic-embed-text
- Perform similarity search against Qdrant with MMR diversity
- Re-rank top-10 results using cross-encoder
- Return top-k context chunks for each sub-query
- Cache retrieved chunks by query hash

**Model:** `nomic-embed-text` (local via Ollama) + `cross-encoder-ms-marco-MiniLM-L-6-v2`

**Inputs:**
```python
{
    "sub_queries": list[str],
    "top_k": int                # Default: 5
}
```

**Outputs:**
```python
{
    "retrieved_chunks": list[dict],  # [{text, source, score, chunk_id}]
    "retrieval_metadata": dict       # Stats: num_queries, total_chunks
}
```

### 3.3 Reasoning Agent

**File:** `agents/reasoning_agent.py`

**Responsibilities:**
- Construct context-augmented prompt from retrieved chunks
- Generate comprehensive answer grounded in retrieved documents
- Cite source chunks used in the answer
- Produce structured response with answer, citations, confidence, reasoning trace

**Model:** Groq `llama3-70b-8192` (primary) → Gemini `gemini-1.5-flash` (fallback)

**Inputs:**
```python
{
    "query": str,
    "sub_queries": list[str],
    "retrieved_chunks": list[dict],
    "chat_history": list[dict]
}
```

**Outputs:**
```python
{
    "answer": str,
    "citations": list[str],       # source IDs
    "confidence": str,            # "low" | "medium" | "high"
    "reasoning_trace": str        # internal chain of thought
}
```

### 3.4 Critic Agent

**File:** `agents/critic_agent.py`

**Responsibilities:**
- Check answer faithfulness: every claim grounded in cited chunks?
- Check answer completeness: addresses all sub-queries?
- Check answer coherence: logically structured?
- Output verdict: `approve`, `retry`, or `escalate`
- If retry: provide specific feedback to Reasoning Agent

**Model:** Groq `llama3-8b-8192`

**Inputs:**
```python
{
    "query": str,
    "retrieved_chunks": list[dict],
    "answer": str,
    "citations": list[str],
    "confidence": str,
    "reasoning_trace": str,
    "retry_count": int
}
```

**Outputs:**
```python
{
    "verdict": str,              # "approve" | "retry" | "escalate"
    "critique": str,             # Feedback for retry
    "final_answer": str | None   # Set only if verdict == "approve"
}
```

---

## 4. Data Flow

### 4.1 LangGraph State Schema

```python
from typing import TypedDict, List, Optional

class RAGState(TypedDict):
    # Input
    query: str
    chat_history: List[dict]

    # Router outputs
    query_type: str
    sub_queries: List[str]
    route: str

    # Retriever outputs
    retrieved_chunks: List[dict]
    retrieval_metadata: dict

    # Reasoning outputs
    answer: str
    citations: List[str]
    confidence: str
    reasoning_trace: str

    # Critic outputs
    verdict: str
    critique: str
    final_answer: Optional[str]

    # Flow control
    retry_count: int
```

### 4.2 Graph Edges

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(RAGState)

# Add nodes
workflow.add_node("router", router_agent)
workflow.add_node("retriever", retriever_agent)
workflow.add_node("reasoning", reasoning_agent)
workflow.add_node("critic", critic_agent)

# Set entry point
workflow.set_entry_point("router")

# Conditional edges from router
workflow.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "retriever": "retriever",
        "reasoner": "reasoning",
        "clarify": END
    }
)

# Standard flow
workflow.add_edge("retriever", "reasoning")
workflow.add_edge("reasoning", "critic")

# Retry loop from critic
workflow.add_conditional_edges(
    "critic",
    lambda state: state["verdict"],
    {
        "approve": END,
        "retry": "reasoning",
        "escalate": END
    }
)

graph = workflow.compile()
```

### 4.3 Provider Fallback Chain

```
Query → Groq llama3-70b
         │ (429/rate limit/timeout)
         ▼
       Gemini 1.5 Flash
         │ (429/rate limit/timeout)
         ▼
       OpenRouter
         │ (429/rate limit/timeout)
         ▼
       Ollama llama3 (local)
```

Each provider:
- Token bucket tracking (max RPM/TPM)
- Exponential backoff: 2^n seconds (n = retry attempt)
- Max 3 retries per provider before switching
- Total max retries: 9 (3 providers × 3 retries + Ollama)

---

## 5. Production Enhancements

### 5.1 Rate Limiting

**File:** `utils/rate_limiter.py`

```python
class RateLimiter:
    def __init__(self, max_rpm: int = 30):
        self.max_rpm = max_rpm
        self.requests = defaultdict(list)  # provider -> [timestamps]

    async def call_with_retry(self, fn: Callable, *args, **kwargs):
        for attempt in range(3):
            if not self._can_make_request(provider):
                await self._backoff(attempt)

            try:
                response = await fn(*args, **kwargs)
                self._record_request(provider)
                return response
            except RateLimitError:
                continue
        raise MaxRetriesExceeded()
```

### 5.2 Semantic Caching

**File:** `utils/cache.py`

Redis-based caching of query embeddings:
- Key: `hash(query_embedding)`
- Value: `retrieved_chunks`
- TTL: 1 hour
- Hit rate target: 60%+ on repeated queries

```python
async def get_cached_chunks(query_embedding: np.ndarray) -> Optional[List[dict]]:
    key = hashlib.sha256(query_embedding.tobytes()).hexdigest()
    cached = await redis.get(f"semantic:{key}")
    return json.loads(cached) if cached else None
```

### 5.3 Response Caching

TTL-based caching of final answers:
- Key: `hash(query + answer)`
- Value: `{answer, citations, confidence}`
- TTL: 24 hours
- Invalidated on document re-ingestion

### 5.4 Graceful Degradation

**Provider Health Check:**
```python
async def check_provider_health(provider: str) -> bool:
    try:
        response = await provider.health_check()
        return response.status == "healthy"
    except:
        return False
```

**Fallback Logic:**
- If Groq unhealthy → try Gemini
- If Gemini unhealthy → try OpenRouter
- If all APIs unhealthy → use Ollama llama3 (local)
- System always responds, never fails completely

### 5.5 Observability

**Structured Logging:**
```python
{
    "timestamp": "2026-04-20T14:30:00Z",
    "trace_id": "abc123",
    "agent": "router",
    "event": "query_classified",
    "query": "What are revenue drivers?",
    "query_type": "factual",
    "latency_ms": 45
}
```

**Metrics Collected:**
- Per-agent latency (p50, p95, p99)
- Cache hit rate (semantic, response)
- Provider success rate
- Retry distribution
- Query classification distribution

---

## 6. Data Ingestion Pipeline

### 6.1 Document Processing

**File:** `rag/ingestor.py`

```python
def ingest_documents(data_dir: str = "./data"):
    """Process all PDFs and populate Qdrant collection."""
    # 1. Load PDFs
    documents = load_pdfs_from_directory(data_dir)

    # 2. Chunk with RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # 3. Generate embeddings via nomic-embed-text
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 4. Create/populate Qdrant collection
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url="http://localhost:6334",
        collection_name="virallens_docs"
    )

    return vector_store
```

### 6.2 Chunking Strategy

**Primary: Recursive Character Text Splitter**
- `chunk_size`: 1000 tokens (balance context richness vs precision)
- `chunk_overlap`: 200 tokens (prevent context loss at boundaries)
- Separators: `["\n\n", "\n", ". ", " ", ""]` (respect natural boundaries)

**Optional: Semantic Chunking**
For structured/technical documents:
- Embedding model: `nomic-embed-text`
- Breakpoint type: Percentile (95th percentile of cosine distance)
- Groups topically coherent sentences

---

## 7. Vector Database: Qdrant

### 7.1 Why Qdrant

| Criterion | Qdrant vs Alternatives |
|-----------|------------------------|
| **Performance** | Rust-based, sub-20ms p99 latency (10x faster than ChromaDB) |
| **Scalability** | Native sharding + replication, horizontal scaling |
| **Features** | Hybrid search, quantization, advanced payload filtering |
| **Operations** | Single-container Docker, health checks, snapshots |
| **Future-Proof** | Add cluster nodes without code changes |

### 7.2 Docker Configuration

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:v1.12.0
    container_name: virallens_qdrant
    ports:
      - "6333:6333"   # gRPC
      - "6334:6334"   # REST API
    volumes:
      - ./qdrant-storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6333
      - QDRANT__SERVICE__HTTP_PORT=6334
      - RUST_LOG=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6334/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
```

### 7.3 Collection Configuration

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6334")

client.recreate_collection(
    collection_name="virallens_docs",
    vectors_config=VectorParams(
        size=768,                    # nomic-embed-text dimension
        distance=Distance.COSINE,
        hnsw_config={
            "m": 16,                 # connectivity parameter
            "ef_construct": 100,     # indexing accuracy
        }
    ),
    optimizers_config={
        "indexing_threshold": 20000 # Build HNSW index after 20K vectors
    }
)
```

---

## 8. Testing Strategy

### 8.1 Unit Tests per Agent

| Agent | Test Coverage | Mocks |
|-------|---------------|-------|
| **Router** | Classification, decomposition, routing | Mock LLM |
| **Retriever** | Embedding, search, MMR, reranking | Mock Qdrant |
| **Reasoning** | Prompt construction, citations | Mock LLM |
| **Critic** | Verdict logic, faithfulness checks | Mock LLM |

**Test Structure:**
```
tests/
├── unit/
│   ├── test_router_agent.py
│   ├── test_retriever_agent.py
│   ├── test_reasoning_agent.py
│   ├── test_critic_agent.py
│   ├── test_rate_limiter.py
│   └── test_cache.py
├── fixtures/
│   ├── sample_chunks.json
│   └── mock_llm_responses.json
└── conftest.py
```

### 8.2 Sample Queries

```python
SAMPLE_QUERIES = [
    {
        "query": "What were the primary revenue drivers in Q3 2024?",
        "expected_type": "factual",
        "description": "Simple factual query requiring retrieval"
    },
    {
        "query": "Compare the performance metrics across all documents",
        "expected_type": "complex",
        "description": "Multi-document comparison requiring decomposition"
    },
    {
        "query": "Hello, can you help me?",
        "expected_type": "conversational",
        "description": "Conversational greeting, no retrieval needed"
    },
    {
        "query": "What's the deal with that thing?",
        "expected_type": "ambiguous",
        "description": "Ambiguous query requiring clarification"
    },
    {
        "query": "Summarize the key findings from the annual report and highlight any risks mentioned",
        "expected_type": "complex",
        "description": "Two-part question: summary + extraction"
    }
]
```

### 8.3 Test Execution

```bash
# Run unit tests
pytest tests/unit/ -v --cov=agents --cov-report=html

# Run sample queries
python main.py --run-samples

# Integration smoke test (end-to-end with local LLM)
python main.py --query "test query" --use-ollama
```

---

## 9. Deliverables Checklist

- [x] Runnable multi-agent RAG system (`graph/`, `agents/`, `utils/`)
- [x] Docker Compose configuration with Qdrant
- [x] Dockerfile for application container
- [x] Data ingestion pipeline (`rag/ingestor.py`)
- [x] README.md with agent roles, LangGraph flow, DeepAgents flow, approach
- [x] Unit tests per agent (`tests/unit/`)
- [x] Sample queries in main.py (5 queries)
- [x] Structured logging with trace IDs
- [x] Multi-provider support (Groq, Gemini, OpenRouter, Ollama)
- [x] Rate limiting with exponential backoff
- [x] Semantic caching (Redis)
- [x] Response caching (TTL-based)
- [x] Graceful degradation to Ollama

---

## 10. Files and Directories

```
q1_rag_orchestration/
├── agents/
│   ├── router_agent.py
│   ├── retriever_agent.py
│   ├── reasoning_agent.py
│   └── critic_agent.py
├── graph/
│   └── workflow.py           # LangGraph StateGraph definition
├── rag/
│   ├── ingestor.py           # Data ingestion pipeline
│   └── chunker.py            # Document chunking strategies
├── utils/
│   ├── rate_limiter.py
│   ├── cache.py
│   ├── logger.py
│   └── prompts.py
├── tests/
│   ├── unit/
│   └── fixtures/
├── main.py                    # Entry point with sample queries
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 11. Alignment with README.txt Requirements

| Requirement | Implementation |
|-------------|----------------|
| LangChain + LangGraph + DeepAgents | ✅ StateGraph with 4 agents, DeepAgents wrapper |
| Use provided files as data | ✅ Data ingestion for 5 PDFs in `data/` |
| Appropriate chunking | ✅ Recursive Character (1000/200) + semantic option |
| Free APIs + rate limiting | ✅ All 4 providers + token bucket + backoff |
| Runnable repo + Dockerfile | ✅ Docker Compose with Qdrant + app |
| README: Agent roles | ✅ Documented in ARCHITECTURE.md / AGENTS.md |
| README: LangGraph + DeepAgents flow | ✅ Documented with diagrams |
| README: Approach | ✅ Chunking, retrieval, MMR, cross-encoder, caching |
| Sample main.py | ✅ 5 representative queries with function calls |

---

**End of Design Spec**
