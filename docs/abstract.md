# Beating RAPTOR on MultiHop-RAG: A Multi-Strategy Approach

## One-Paragraph Summary

We built a RAG system that achieves **74.3% Recall@10** on MultiHop-RAG, surpassing RAPTOR's ~70%. Our key insight: **reranking matters more than retrieval strategy selection**. By combining RAPTOR's hierarchical tree with Cohere's neural reranker (`rerank-english-v3.0`), BM25 hybrid search, and query decomposition, we improved recall by 19 percentage points over vanilla dense retrieval. The single biggest contributor was Cohere reranking (+9.3%), suggesting that for multi-hop QA, precision-focused reranking is more impactful than recall-focused retrieval diversity.

## Key Numbers

| Metric | Value |
|--------|-------|
| **Recall@10** | 74.3% |
| **Improvement over RAPTOR** | +4% |
| **Improvement over dense** | +19% |
| **Latency** | ~1s/query |
| **Cost** | ~$0.003/query |

## The Stack

```
RAPTOR Hierarchy + HyDE + BM25 + Query Decomposition
                    ↓
            Cohere Reranking
                    ↓
               Top-10 Results
```

## Main Takeaways

1. **Add a neural reranker first** — it's the highest-ROI improvement
2. **Hybrid search (BM25 + dense) catches edge cases** — don't skip it
3. **More strategies ≠ better** — 4-5 is the sweet spot before diminishing returns
4. **Latency matters** — a fast, complete result beats a slow, partial one

## Links

- 📝 [Full Blog Post](./blog_post.md)
- 📊 [Technical Report](./technical_report.md)
- 💻 [Code: incidentfox/ultimate_rag](https://github.com/incidentfox/incidentfox)

---

*February 2026 • Built with Claude as AI pair programmer*
