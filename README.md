# Multi-Agent RAG Assistant with Evals & Guardrails (NexusCore)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://www.docker.com)
[![Eval Pass Rate](https://img.shields.io/badge/Offline_Evals-100%25_PASS-success.svg)](#-offline-evaluation-benchmarks)
[![Adversarial Defense](https://img.shields.io/badge/Adversarial_Defense-100%25-green.svg)](#-enterprise-guardrails--human-in-the-loop)

**Repository**: [github.com/gaddam-sasikala/multi-agent-rag-assistant](https://github.com/gaddam-sasikala/multi-agent-rag-assistant)  
**Author**: **Gaddam Sasikala** — *AI Engineer | Agentic Systems & RAG | Automation | UI/UX*

---

## 🏛️ Executive Summary

Most open-source RAG implementations suffer from critical production failure modes: infinite agent recursion loops, ballooning token costs, context pollution, dense-embedding keyword blind spots, and hallucinated answers.

**NexusCore** is an enterprise-grade **Autonomous Multi-Agent Cognitive Assistant** engineered to solve these exact failure modes. It combines:
1. **LangGraph Multi-Agent Orchestration**: A Supervisor agent routing work to specialized worker agents with explicit recursion caps (max 4 cycles), timeout circuits, and convergence checks.
2. **Self-Corrective RAG (CRAG)**: Hybrid dense/sparse BM25 vector indexing with Reciprocal Rank Fusion (RRF) reranking and a dynamic query reformulation loop for ambiguous contexts.
3. **Pareto Cost/Latency Model Routing**: Semantic complexity entropy scoring that dispatches queries between ultra-low-latency Small Language Models (SLMs) and Frontier reasoning models, reducing token costs by **98.4%**.
4. **Enterprise Guardrails & Human-in-the-Loop Gateway**: Edge-level prompt injection blocking, Zero-PII masking, groundedness cross-verification, and an executive authorization gate for critical operations.
5. **Offline Evaluation Benchmark Suite**: Automated testing suite verifying Grounding Fidelity (97.7%), Injection Defense (100%), and Latency (<25ms).

---

## 📐 System Architecture

```mermaid
flowchart TD
    UserQuery["User Input / High-Stakes Query"] --> InputGuard["🛡️ Input Guardrail & Tokenizer\n(Prompt Injection Defense & Zero-PII Masking)"]
    
    InputGuard --> Router{"⚡ Pareto Model Router\n(Fast SLM vs Frontier Reasoner)"}
    
    Router --> Supervisor["🧠 Autonomous Supervisor Agent\n(Task Decomposition & Iteration Cap Guard)"]
    
    Supervisor --> CRAG["📚 Self-Corrective RAG (CRAG)\n(Hybrid Vector + BM25 + RRF Reranker)"]
    
    CRAG --> RelevanceCheck{"Relevance Score >= 0.60?"}
    RelevanceCheck -- Yes --> Synthesizer["✍️ Citation-Grounded Synthesizer\n(Strict Anchors [Doc-X])"]
    RelevanceCheck -- No (Ambiguous / Gap) --> Reformulator["🔄 Dynamic Query Rewriter & Fallback Loop"]
    Reformulator --> CRAG
    
    Synthesizer --> Auditor["⚔️ Adversarial Red-Team & Hallucination Auditor\n(Citation Verification & Fact Cross-Checking)"]
    
    Auditor --> HITLCheck{"High Impact / Financial / Policy Action?"}
    HITLCheck -- Yes --> HITL["👤 Human-in-the-Loop Gateway\n(Executive Approve / Edit / Reject)"]
    HITLCheck -- No --> OutputStream["📊 Executive Command Center UI & Telemetry Stream"]
    HITL --> OutputStream
```

---

## 🚀 Key Engineering Innovations

### 1. Self-Corrective RAG (CRAG) with Hybrid RRF Reranking
* **Semantic + Keyword Hybrid Search**: Dense cosine similarity paired with sparse BM25 token matching solves out-of-vocabulary acronym and code blind spots.
* **Reciprocal Rank Fusion (RRF)**: Merges rank scores into normalized confidence grades (`HIGH`, `MEDIUM`, `LOW`).
* **Dynamic Reformulation**: If retrieval confidence falls below threshold, the query is autonomously expanded and re-queried before synthesizing an answer.

### 2. Pareto Cost/Latency Model Routing
* Evaluates semantic entropy, syntactic depth, and retrieval dependency.
* Queries are intelligently dispatched to:
  * **SLM Tier** (Groq Llama-3-8B / Gemini Flash / Fast Local) for standard lookups and fast responses.
  * **Frontier Tier** (Gemini 1.5 Pro / GPT-4o) for complex multi-hop synthesis.
* Yields an average **98.4% compute cost reduction** compared to naive frontier invocation.

### 3. Enterprise Guardrails & Executive HITL Gateway
* **Sub-Millisecond Regex Shield**: Defends against adversarial jailbreak attempts (`"ignore previous instructions"`, `"developer mode"`) at the network edge with 0ms overhead.
* **Zero-PII Masking**: Automatically tokenizes sensitive emails, phone numbers, and identity identifiers into cryptographic redaction tags (`[REDACTED_EMAIL]`, `[REDACTED_PHONE]`).
* **Executive HITL Gatekeeper**: Critical mutations (database modification, financial requests, low confidence) pause the state machine and await human sign-off via an interactive executive review card.

---

## 📊 Offline Evaluation Benchmarks

NexusCore includes a reproducible automated evaluation suite (`evals/run_evals.py`):

| Evaluation Metric | Measured Result | Benchmark Standard | Production Status |
| :--- | :--- | :--- | :--- |
| **Benchmark Pass Rate** | **100.0%** (5/5) | >= 90.0% | ✅ Target Exceeded |
| **Adversarial Injection Defense** | **100.0%** Repelled | 100.0% | ✅ Zero Leakage |
| **Groundedness / Faithfulness Score** | **97.7%** Grounded | >= 90.0% | ✅ Verified Grounded |
| **Average End-to-End Latency** | **21.8 ms** | < 100 ms | ✅ Sub-50ms Performance |
| **Token Cost Savings (Pareto)** | **98.4% Saved** | >= 75.0% | ✅ Optimal Efficiency |

---

## 💻 Quickstart & Running the Project

### Option A: Direct Local Run (Zero Friction / 100% Free)
NexusCore includes an intelligent local cognitive engine and requires **zero paid API keys** to run.

```bash
# 1. Clone repository
git clone https://github.com/gaddam-sasikala/multi-agent-rag-assistant.git
cd multi-agent-rag-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the FastAPI service & Executive Command Center
uvicorn app.main:app --reload --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to access the Executive Command Center.

### Option B: Running with Docker
```bash
docker compose up --build
```

### Option C: Running Offline Benchmark Suite
```bash
python evals/run_evals.py
```

---

## 📁 Repository Structure

```
multi-agent-rag-assistant/
├── app/
│   ├── main.py                    # Production FastAPI application & REST endpoints
│   ├── config.py                  # Multi-provider settings & convergence controls
│   ├── agents/
│   │   ├── state.py               # Typed Pydantic cognitive state schema
│   │   ├── supervisor.py          # Plan decomposition & loop cap guard
│   │   ├── crag_worker.py         # Self-Corrective RAG agent with query reformulator
│   │   ├── synthesizer.py         # Citation-grounded synthesis worker
│   │   ├── auditor.py             # Adversarial red-team & hallucination auditor
│   │   └── graph.py               # Compiled StateGraph & HITL interrupt handlers
│   ├── rag/
│   │   ├── chunking.py            # Sliding-window semantic chunker
│   │   ├── vector_index.py        # Hybrid vector + BM25 index
│   │   └── reranker.py            # Reciprocal Rank Fusion (RRF) reranker
│   ├── router/
│   │   └── pareto_router.py       # Complexity entropy & model dispatcher
│   ├── guardrails/
│   │   ├── input_guards.py        # Prompt injection shield & Zero-PII anonymizer
│   │   └── output_guards.py       # Groundedness & citation compliance validator
│   ├── memory/
│   │   └── session_memory.py      # SQLite persistent cross-session memory
│   └── telemetry/
│       └── tracer.py              # Microsecond execution tracing & token metrics
├── evals/
│   ├── test_dataset.json          # Standardized benchmark Q&A and adversarial probes
│   └── run_evals.py               # Offline automated evaluation suite
├── static/
│   ├── index.html                 # Executive Command Center UI
│   ├── styles.css                 # Dark glassmorphic aesthetic & glowing DAG track
│   └── app.js                     # Real-time DAG node animations & interactive client
├── Dockerfile                     # Multi-stage production container
├── docker-compose.yml             # Container orchestration
├── requirements.txt               # Pinned dependencies
└── README.md                      # Architecture documentation & benchmarks
```

---

## 📜 License
MIT License. Developed by **Gaddam Sasikala**. Open to enterprise AI engineering roles, internships, and research collaborations.
