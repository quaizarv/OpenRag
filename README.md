# RAG Benchmarking: Multi-Strategy Retrieval for Multi-Hop QA

[![Recall@10](https://img.shields.io/badge/MultiHop--RAG-72.89%25-brightgreen)](./docs/technical_report.md)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

A complete RAG system that achieves **72.89% Recall@10** on MultiHop-RAG, surpassing RAPTOR's ~70%. This repository includes:

- 🔧 **Full RAG Implementation** (`ultimate_rag/`) - RAPTOR + Graph + HyDE + BM25 + Neural Reranking
- 📊 **Benchmark Suite** (`adapters/`, `scripts/`) - Evaluation harness for MultiHop-RAG, CRAG
- 📝 **Documentation** (`docs/`) - Blog post, technical report, architecture

---

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repo
git clone https://github.com/incidentfox/OpenRag.git
cd rag_benchmarking

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
export OPENAI_API_KEY="sk-..."
export COHERE_API_KEY="..."  # Optional but recommended for best performance
```

### 3. Start the RAG Server

```bash
cd ultimate_rag
python -m api.server
```

Server runs at `http://localhost:8000`. Check health: `curl http://localhost:8000/health`

### 4. Run Benchmark

```bash
# MultiHop-RAG (2556 queries)
python scripts/run_multihop_eval.py --queries 100  # Quick test

# Full benchmark
python scripts/run_multihop_eval.py
```

---

## Results

| Benchmark | Queries Tested | Our Result | SOTA | Notes |
|-----------|----------------|------------|------|-------|
| **MultiHop-RAG** | 2,556 (full) | **72.89%** | ~70% | Beats RAPTOR baseline |
| **SQuAD** | 200+ (ongoing) | **99.0%** | ~85-90% | Full benchmark running on EC2 |
| **CRAG** | 10 (sample) | **70%** | ~50-60% | Per-query corpus test |

> **Note on SQuAD:** Full 10,570-query benchmark running on EC2. After 200 queries: 99.0% Recall@10.

> **Note on CRAG:** Tested 10 queries using each query's provided search results as corpus. Scaling requires per-query ingestion which is compute-intensive. CRAG is designed for API-augmented RAG, not static document retrieval.

### Ablation Study

| Component | Recall@10 | Δ from baseline |
|-----------|-----------|-----------------|
| Semantic only | 55.2% | — |
| + RAPTOR hierarchy | 62.5% | +7.3% |
| + Cohere reranking | 71.8% | +16.6% |
| + BM25 hybrid | 72.4% | +17.2% |
| + HyDE + Query decomp | 72.89% | +17.7% |

**Key insight:** Cohere's neural reranker alone adds +9.3 percentage points.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Query Input                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Parallel Retrieval Strategies                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Semantic │ │   HyDE   │ │   BM25   │ │  Query   │           │
│  │  Search  │ │ Expansion│ │  Hybrid  │ │  Decomp  │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Cohere Neural Reranking                       │
│                  (rerank-english-v3.0)                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Top-K Results                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
rag_benchmarking/
├── ultimate_rag/              # 🔧 Full RAG implementation
│   ├── api/
│   │   └── server.py          # FastAPI server
│   ├── retrieval/
│   │   ├── retriever.py       # Main orchestration
│   │   ├── strategies.py      # HyDE, BM25, decomposition
│   │   └── reranker.py        # Cohere + cross-encoder
│   ├── raptor/
│   │   └── tree_building.py   # RAPTOR hierarchy
│   ├── graph/
│   │   └── graph.py           # Knowledge graph
│   ├── core/
│   │   └── node.py            # Tree/forest data structures
│   └── agents/
│       └── teaching.py        # Knowledge teaching interface
│
├── knowledge_base/            # 📚 RAPTOR core library
│   └── raptor/
│       ├── cluster_tree_builder.py
│       ├── EmbeddingModels.py
│       └── ...
│
├── adapters/                  # 🔌 Benchmark adapters
│   └── ultimate_rag_adapter.py
│
├── scripts/                   # 🚀 Evaluation scripts
│   ├── run_multihop_eval.py
│   └── run_crag_eval.py
│
├── docs/                      # 📝 Documentation
│   ├── blog_post.md           # Practitioner-friendly writeup
│   ├── technical_report.md    # Academic-style report
│   └── README.md
│
├── multihop_rag/              # 📊 MultiHop-RAG dataset
│   └── dataset/
│       ├── corpus.json        # 609 news articles
│       └── MultiHopRAG.json   # 2556 queries
│
├── crag/                      # 📊 CRAG dataset
│   └── ...
│
└── requirements.txt           # Dependencies
```

---

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Query (Retrieval)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the outcome of the merger?", "top_k": 10}'
```

### Ingest Documents
```bash
curl -X POST http://localhost:8000/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{
    "tree": "default",
    "documents": [{"content": "Document text here..."}],
    "build_hierarchy": true
  }'
```

### Save/Load Tree
```bash
# Save
curl -X POST http://localhost:8000/persist/save \
  -H "Content-Type: application/json" \
  -d '{"tree": "default"}'

# Load
curl -X POST http://localhost:8000/persist/load \
  -H "Content-Type: application/json" \
  -d '{"tree": "default", "path": "trees/default.pkl"}'
```

---

## Configuration

### Retrieval Modes

| Mode | Strategies | Use Case |
|------|------------|----------|
| `fast` | Semantic only | Low latency, simple queries |
| `standard` | Semantic + HyDE + BM25 + Decomp | Balanced (default) |
| `thorough` | All strategies | Maximum recall, high latency |

### Environment Variables

```bash
OPENAI_API_KEY=sk-...          # Required for embeddings
COHERE_API_KEY=...             # Recommended for reranking (see privacy note below)
RETRIEVAL_MODE=standard        # fast|standard|thorough
DEFAULT_TOP_K=10               # Number of results
```

### Privacy Notice: Cohere Reranker

This system uses [Cohere's rerank API](https://cohere.com/rerank) for neural reranking, which provides the best benchmark results (+9.3% improvement). **Please be aware:**

- **Data logging:** By default, Cohere logs prompts and outputs on their SaaS platform (retained for 30 days)
- **Training opt-out:** You can disable data usage for training in your [Cohere dashboard](https://dashboard.cohere.com/) under "Data Controls"
- **Zero retention:** Enterprise customers can request zero data retention
- **Cloud deployments:** If using Cohere via AWS/GCP/Azure, Cohere does not receive your data

**For privacy-sensitive use cases**, consider these alternatives:

1. **Local cross-encoder:** The system includes `CrossEncoderReranker` using `BAAI/bge-reranker-base` (runs locally, no external API)
2. **Remove Cohere:** Don't set `COHERE_API_KEY` and the system falls back to local reranking
3. **LLM-as-reranker:** Use a local/GDPR-compliant LLM for reranking

See [Cohere's privacy policy](https://cohere.com/privacy) and [enterprise data commitments](https://cohere.com/enterprise-data-commitments) for details.

---

## Cost Analysis

| Component | Cost per Query |
|-----------|----------------|
| OpenAI embeddings | $0.000007 |
| HyDE generation | $0.00018 |
| Query decomposition | $0.00027 |
| Cohere reranking | $0.002 |
| **Total** | **~$0.0025** |

Full benchmark (2556 queries): **~$6**

---

## Documentation

- 📝 [Blog Post](./docs/blog_post.md) - Practitioner-friendly writeup
- 📊 [Technical Report](./docs/technical_report.md) - Detailed analysis with ablations
- 🏗️ [Architecture](./ultimate_rag/ARCHITECTURE.md) - System design

---

## Citation

If you use this code, please cite:

```bibtex
@software{rag_benchmarking_2026,
  title = {Multi-Strategy RAG for Multi-Hop Question Answering},
  author = {Anonymous},
  year = {2026},
  url = {https://github.com/incidentfox/OpenRag}
}
```

---

## License

MIT License - see [LICENSE](./LICENSE) for details.

---

## Acknowledgments

- [RAPTOR](https://arxiv.org/abs/2401.18059) for hierarchical retrieval
- [Cohere](https://cohere.com) for neural reranking API
- [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) for benchmark dataset
- Built with [Claude](https://anthropic.com) as AI pair programmer
