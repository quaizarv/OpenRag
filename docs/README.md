# Documentation

This folder contains documentation for the RAG Benchmarking project.

## Contents

### [Blog Post](./blog_post.md)
A practitioner-friendly writeup explaining our approach to achieving 72.89% Recall@10 on MultiHop-RAG. Covers:
- Problem motivation
- System architecture
- Component deep dives
- Lessons learned
- Cost analysis

**Best for:** Sharing with the RAG community, LinkedIn, technical blog

### [Technical Report](./technical_report.md)
A detailed technical report with:
- Full system architecture
- Algorithm descriptions with pseudocode
- Ablation studies
- Failure analysis
- Hyperparameter sensitivity
- References

**Best for:** Academic audiences, internal documentation, reproducibility

## Quick Stats

| Benchmark | Queries | Our Result | SOTA |
|-----------|---------|------------|------|
| MultiHop-RAG | 2,556 | 72.89% Recall@10 | ~70% |
| CRAG | 2,706 | (running) | TBD |

## Key Findings

1. **Cohere reranking is the biggest win:** +9.3% over base system
2. **BM25 hybrid still matters:** Keywords catch what embeddings miss
3. **More strategies have diminishing returns:** 4-5 is the sweet spot
4. **Cost is manageable:** ~$0.003/query with full pipeline

## System Components

```
┌─────────────────────────────────────────┐
│              Query Input                 │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         Parallel Strategies              │
│  • Semantic Search                       │
│  • HyDE Expansion                        │
│  • BM25 Hybrid                           │
│  • Query Decomposition                   │
│  • Graph Traversal                       │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         Cohere Reranking                 │
│      (rerank-english-v3.0)               │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│           Top-K Results                  │
└─────────────────────────────────────────┘
```

## Related Code

- **RAG System:** [`incidentfox/ultimate_rag/`](../../incidentfox/ultimate_rag/)
- **Benchmark Adapter:** [`adapters/ultimate_rag_adapter.py`](../adapters/ultimate_rag_adapter.py)
- **Evaluation Scripts:** [`scripts/`](../scripts/)

## Authors

Built using Claude (Anthropic) as AI pair programmer, February 2026.
