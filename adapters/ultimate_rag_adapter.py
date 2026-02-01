"""
Ultimate RAG Adapter for Benchmark Evaluation.

This adapter connects the Ultimate RAG system to standard RAG benchmarks,
allowing evaluation against MultiHop-RAG, RAGBench, CRAG, and others.
"""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class RetrievalResult:
    """A single retrieved chunk."""
    text: str
    score: float
    importance: float = 0.0
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of running a benchmark query."""
    query: str
    retrieved_chunks: List[RetrievalResult]
    generated_answer: Optional[str] = None
    ground_truth: Optional[str] = None
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    strategies_used: List[str] = field(default_factory=list)


class UltimateRAGAdapter:
    """
    Adapter to connect Ultimate RAG to standard RAG benchmarks.

    This adapter provides a unified interface for:
    1. Ingesting benchmark documents into Ultimate RAG
    2. Running retrieval queries
    3. Generating answers using retrieved context
    4. Evaluating results against ground truth
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        timeout: int = 60,  # Increased for complex queries
        default_top_k: int = 5,
        retrieval_mode: str = "thorough",  # 'fast' for speed, 'standard' for quality, 'thorough' for best (slow)
    ):
        """
        Initialize the adapter.

        Args:
            api_url: Base URL of the Ultimate RAG API server
            timeout: Request timeout in seconds
            default_top_k: Default number of chunks to retrieve
            retrieval_mode: Default retrieval mode (standard, fast, thorough, incident)
        """
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.default_top_k = default_top_k
        self.retrieval_mode = retrieval_mode
        self._session = requests.Session()

    def health_check(self) -> bool:
        """Check if the Ultimate RAG server is healthy."""
        try:
            resp = self._session.get(
                f"{self.api_url}/health",
                timeout=self.timeout
            )
            return resp.status_code == 200 and resp.json().get("status") == "healthy"
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def ingest_document(
        self,
        content: str,
        source_url: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document into Ultimate RAG.

        Args:
            content: Document content
            source_url: Optional source URL
            content_type: Optional content type (markdown, text, html, etc.)
            metadata: Optional additional metadata

        Returns:
            Ingestion response with chunks_created, entities_found, etc.
        """
        payload = {
            "content": content,
            "source_url": source_url,
            "content_type": content_type,
            "metadata": metadata or {},
        }

        resp = self._session.post(
            f"{self.api_url}/ingest",
            json=payload,
            timeout=self.timeout * 2,  # Ingestion can take longer
        )
        resp.raise_for_status()
        return resp.json()

    def ingest_benchmark_corpus(
        self,
        documents: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None,
        batch_size: int = 50,
        build_hierarchy: bool = False,
        hierarchy_num_layers: int = 5,
        hierarchy_target_top_nodes: int = 50,
    ) -> Dict[str, Any]:
        """
        Ingest an entire benchmark corpus using batch endpoint.

        Args:
            documents: List of documents, each with 'content' and optional metadata
            progress_callback: Optional callback(current, total) for progress
            batch_size: Number of documents per batch (default 50)
            build_hierarchy: Whether to build full RAPTOR hierarchy (clustering + summarization)
            hierarchy_num_layers: Max tree layers when building hierarchy (default 5)
            hierarchy_target_top_nodes: Target size for top layer (default 50)

        Returns:
            Summary of ingestion results
        """
        results = {
            "total": len(documents),
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "total_entities": 0,
            "total_nodes": 0,
            "num_layers": 0,
            "layer_distribution": {},
            "errors": [],
        }

        # When building hierarchy, we should send ALL documents in one batch
        # because RAPTOR needs the full corpus to build the tree structure
        if build_hierarchy:
            print(f"Building RAPTOR hierarchy from {len(documents)} documents...")
            print(f"  Config: layers={hierarchy_num_layers}, target_top={hierarchy_target_top_nodes}")

            # Prepare all documents
            all_docs = []
            for i, doc in enumerate(documents):
                content = doc.get("content", doc.get("text", doc.get("body", "")))
                source = doc.get("source", doc.get("url", doc.get("id", f"doc_{i}")))
                metadata = doc.get("metadata", {})
                if "title" in doc and "title" not in metadata:
                    metadata["title"] = doc["title"]
                all_docs.append({
                    "content": content,
                    "source_url": source,
                    "metadata": metadata,
                })

            try:
                payload = {
                    "documents": all_docs,
                    "build_hierarchy": True,
                    "hierarchy_num_layers": hierarchy_num_layers,
                    "hierarchy_target_top_nodes": hierarchy_target_top_nodes,
                }

                # Hierarchy building can take a LONG time (minutes to hours)
                # Use a very long timeout
                resp = self._session.post(
                    f"{self.api_url}/ingest/batch",
                    json=payload,
                    timeout=3600,  # 1 hour timeout for hierarchy building
                )
                resp.raise_for_status()
                result = resp.json()

                if result.get("success"):
                    results["successful"] = len(documents)
                    results["total_chunks"] = result.get("total_chunks", 0)
                    results["total_nodes"] = result.get("total_nodes_created", 0)
                    results["num_layers"] = result.get("num_layers", 0)
                    results["layer_distribution"] = result.get("layer_distribution", {})
                    results["total_entities"] = len(result.get("entities_found", []))
                    
                    print(f"  ✓ Built {results['num_layers']}-layer tree with {results['total_nodes']} total nodes")
                    for layer, count in sorted(results["layer_distribution"].items()):
                        print(f"    Layer {layer}: {count} nodes")
                else:
                    results["failed"] = len(documents)
                    results["errors"].append(f"Hierarchy build failed: {result.get('warnings', [])}")
                    print(f"  ✗ Build failed: {result.get('warnings', [])}")

                if progress_callback:
                    progress_callback(len(documents), len(documents))

            except Exception as e:
                results["failed"] = len(documents)
                results["errors"].append(f"Hierarchy build error: {str(e)}")
                print(f"  ✗ Error: {e}")

        else:
            # Standard flat batched ingestion (existing behavior)
            for batch_start in range(0, len(documents), batch_size):
                batch_end = min(batch_start + batch_size, len(documents))
                batch = documents[batch_start:batch_end]

                # Prepare batch payload
                batch_docs = []
                for i, doc in enumerate(batch):
                    content = doc.get("content", doc.get("text", doc.get("body", "")))
                    source = doc.get("source", doc.get("url", doc.get("id", f"doc_{batch_start + i}")))
                    metadata = doc.get("metadata", {})
                    
                    # Include title in metadata if available
                    if "title" in doc and "title" not in metadata:
                        metadata["title"] = doc["title"]

                    batch_docs.append({
                        "content": content,
                        "source_url": source,
                        "metadata": metadata,
                    })

                try:
                    # Use batch ingest endpoint
                    resp = self._session.post(
                        f"{self.api_url}/ingest/batch",
                        json={"documents": batch_docs},
                        timeout=self.timeout * 10,  # Batch can take longer
                    )
                    resp.raise_for_status()
                    result = resp.json()

                    if result.get("success"):
                        results["successful"] += len(batch)
                        results["total_chunks"] += result.get("total_chunks", 0)
                        results["total_entities"] += len(result.get("entities_found", []))
                    else:
                        results["failed"] += len(batch)
                        results["errors"].append(f"Batch {batch_start}-{batch_end}: {result.get('warnings', [])}")

                except Exception as e:
                    # Fallback to individual ingestion if batch fails
                    results["errors"].append(f"Batch {batch_start}-{batch_end} failed: {str(e)}, falling back to individual")
                    for i, doc in enumerate(batch):
                        try:
                            content = doc.get("content", doc.get("text", doc.get("body", "")))
                            source = doc.get("source", doc.get("url", doc.get("id", f"doc_{batch_start + i}")))
                            result = self.ingest_document(content=content, source_url=source)
                            if result.get("success"):
                                results["successful"] += 1
                                results["total_chunks"] += result.get("chunks_created", 0)
                            else:
                                results["failed"] += 1
                        except Exception as e2:
                            results["failed"] += 1
                            results["errors"].append(f"Doc {batch_start + i}: {str(e2)}")

                if progress_callback:
                    progress_callback(batch_end, len(documents))

        # Auto-save after ingestion to persist data
        if results["successful"] > 0:
            try:
                save_result = self.save_tree()
                if save_result.get("success"):
                    results["persisted"] = True
                    results["persist_path"] = save_result.get("local_paths", [])
            except Exception as e:
                results["persist_error"] = str(e)

        return results

    def save_tree(self, tree_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Save tree(s) to disk for persistence.

        Args:
            tree_name: Specific tree to save, or None for all trees

        Returns:
            Save result with paths
        """
        payload = {"to_local": True, "to_s3": False}
        if tree_name:
            payload["tree"] = tree_name

        resp = self._session.post(
            f"{self.api_url}/persist/save",
            json=payload,
            timeout=self.timeout * 5,
        )
        resp.raise_for_status()
        return resp.json()

    def load_tree(self, tree_name: str) -> Dict[str, Any]:
        """
        Load a tree from disk.

        Args:
            tree_name: Name of tree to load

        Returns:
            Load result
        """
        resp = self._session.post(
            f"{self.api_url}/persist/load",
            json={"tree": tree_name, "from_s3": False},
            timeout=self.timeout * 5,
        )
        resp.raise_for_status()
        return resp.json()

    def list_available_trees(self) -> Dict[str, Any]:
        """
        List trees available for loading.

        Returns:
            Dict with local_trees, s3_trees, loaded_trees
        """
        resp = self._session.get(
            f"{self.api_url}/persist/available",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_graph: bool = True,
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The search query
            top_k: Number of chunks to retrieve
            mode: Retrieval mode (standard, fast, thorough, incident)
            filters: Optional filters
            include_graph: Whether to include graph context

        Returns:
            Tuple of (list of RetrievalResult, metadata dict)
        """
        start_time = time.time()

        payload = {
            "query": query,
            "top_k": top_k or self.default_top_k,
            "mode": mode or self.retrieval_mode,
            "filters": filters,
            "include_graph": include_graph,
        }

        resp = self._session.post(
            f"{self.api_url}/query",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        elapsed_ms = (time.time() - start_time) * 1000

        results = [
            RetrievalResult(
                text=r["text"],
                score=r["score"],
                importance=r.get("importance", 0.0),
                source=r.get("source"),
                metadata=r.get("metadata", {}),
            )
            for r in data.get("results", [])
        ]

        metadata = {
            "retrieval_time_ms": elapsed_ms,
            "total_candidates": data.get("total_candidates", 0),
            "mode": data.get("mode", ""),
            "strategies_used": data.get("strategies_used", []),
        }

        return results, metadata

    def retrieve_for_benchmark(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Retrieve for benchmark evaluation - returns BenchmarkResult.

        Args:
            query: The search query
            top_k: Number of chunks to retrieve

        Returns:
            BenchmarkResult with retrieved chunks
        """
        results, metadata = self.retrieve(query, top_k=top_k)

        return BenchmarkResult(
            query=query,
            retrieved_chunks=results,
            retrieval_time_ms=metadata["retrieval_time_ms"],
            strategies_used=metadata["strategies_used"],
        )

    def generate_answer(
        self,
        query: str,
        context_chunks: List[str],
        max_tokens: int = 500,
    ) -> str:
        """
        Generate an answer using retrieved context.

        Note: This uses the v1 /answer endpoint. For more control,
        you may want to use your own LLM with the retrieved context.

        Args:
            query: The question
            context_chunks: Retrieved context chunks
            max_tokens: Maximum tokens in response

        Returns:
            Generated answer string
        """
        # Use v1 answer endpoint
        payload = {
            "question": query,
            "top_k": len(context_chunks),
        }

        resp = self._session.post(
            f"{self.api_url}/api/v1/answer",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("answer", "")

    def retrieve_and_generate(
        self,
        query: str,
        top_k: Optional[int] = None,
        ground_truth: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        Full RAG pipeline: retrieve then generate.

        Args:
            query: The question
            top_k: Number of chunks to retrieve
            ground_truth: Optional ground truth answer for evaluation

        Returns:
            BenchmarkResult with retrieval and generation results
        """
        # Retrieve
        results, retrieval_meta = self.retrieve(query, top_k=top_k)

        # Generate
        gen_start = time.time()
        context_chunks = [r.text for r in results]
        answer = self.generate_answer(query, context_chunks)
        gen_time_ms = (time.time() - gen_start) * 1000

        return BenchmarkResult(
            query=query,
            retrieved_chunks=results,
            generated_answer=answer,
            ground_truth=ground_truth,
            retrieval_time_ms=retrieval_meta["retrieval_time_ms"],
            generation_time_ms=gen_time_ms,
            strategies_used=retrieval_meta["strategies_used"],
        )

    def get_context_string(self, results: List[RetrievalResult]) -> str:
        """
        Format retrieved results as a context string for LLM input.

        Args:
            results: List of RetrievalResult

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, r in enumerate(results, 1):
            source_info = f" (source: {r.source})" if r.source else ""
            context_parts.append(f"[{i}]{source_info}\n{r.text}")

        return "\n\n---\n\n".join(context_parts)


class MultiHopRAGEvaluator:
    """
    Evaluator for the MultiHop-RAG benchmark.

    This handles the specific format and metrics of MultiHop-RAG.
    """

    def __init__(self, adapter: UltimateRAGAdapter, data_dir: str):
        self.adapter = adapter
        self.data_dir = Path(data_dir)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the MultiHop-RAG dataset."""
        dataset_path = self.data_dir / "dataset"

        # Try to load from common locations
        for filename in ["MultiHopRAG.json", "dataset.json", "queries.json"]:
            path = dataset_path / filename
            if path.exists():
                with open(path) as f:
                    return json.load(f)

        raise FileNotFoundError(f"Could not find dataset in {dataset_path}")

    def load_corpus(self) -> List[Dict[str, Any]]:
        """Load the document corpus."""
        dataset_path = self.data_dir / "dataset"

        # Try corpus.json file first (MultiHop-RAG format)
        corpus_file = dataset_path / "corpus.json"
        if corpus_file.exists():
            with open(corpus_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    return [data]

        # Try corpus directory (alternative format)
        corpus_dir = dataset_path / "corpus"
        documents = []
        if corpus_dir.exists() and corpus_dir.is_dir():
            for json_file in corpus_dir.glob("*.json"):
                with open(json_file) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        documents.extend(data)
                    else:
                        documents.append(data)

        return documents

    def evaluate_retrieval(
        self,
        queries: List[Dict[str, Any]],
        top_k: int = 5,
        max_workers: int = 10,
        progress_callback: Optional[callable] = None,
        output_file: Optional[str] = None,
        save_interval: int = 100,
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval performance with concurrent execution.

        Args:
            queries: List of query dicts with 'query' and 'evidence_list'
            top_k: Number of chunks to retrieve
            max_workers: Number of concurrent workers (default 10)
            progress_callback: Optional callback(current, total) for progress
            output_file: Optional path to save intermediate results
            save_interval: Save results every N queries (default 100)

        Returns:
            Metrics dict with recall, MRR, hit_rate, etc.
        """
        results = {
            "total_queries": len(queries),
            "recall_at_k": [],
            "mrr": [],
            "hit_rate": 0,
            "avg_retrieval_time_ms": 0,
        }

        def process_query(q: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single query and return metrics."""
            query_text = q.get("query", q.get("question", ""))
            # MultiHop-RAG uses 'evidences' key with structured evidence objects
            evidence_list = q.get("evidences", q.get("evidence_list", q.get("supporting_facts", [])))

            # Retrieve
            benchmark_result = self.adapter.retrieve_for_benchmark(query_text, top_k=top_k)

            # Get retrieved texts - normalize like official evaluation
            # (remove spaces and newlines for robust matching)
            def normalize(text: str) -> str:
                return text.replace(" ", "").replace("\n", "").lower()
            
            retrieved_texts = [normalize(r.text) for r in benchmark_result.retrieved_chunks]

            # Calculate recall
            recall = 0.0
            mrr = 0.0
            hit = False
            skipped = False  # Mark queries with no evidence

            if not evidence_list:
                # No evidence to evaluate - skip this query
                skipped = True
            elif evidence_list:
                found = 0
                first_rank = None
                for i, evidence in enumerate(evidence_list):
                    # Handle structured evidence objects (MultiHop-RAG format)
                    if isinstance(evidence, dict):
                        # Try to match by title or fact content (normalized)
                        title = normalize(evidence.get("title", ""))
                        fact = normalize(evidence.get("fact", ""))
                        body = normalize(evidence.get("body", ""))

                        for rank, retrieved in enumerate(retrieved_texts):
                            # Check if title or fact snippet is in retrieved text
                            if (title and title in retrieved) or \
                               (fact and fact in retrieved) or \
                               (body and body in retrieved):
                                found += 1
                                if first_rank is None:
                                    first_rank = rank + 1
                                break
                    else:
                        # Plain text evidence
                        evidence_text = normalize(str(evidence))
                        for rank, retrieved in enumerate(retrieved_texts):
                            if evidence_text in retrieved or retrieved in evidence_text:
                                found += 1
                                if first_rank is None:
                                    first_rank = rank + 1
                                break

                recall = found / len(evidence_list) if evidence_list else 0

                if first_rank:
                    mrr = 1.0 / first_rank
                    hit = True

            return {
                "recall": recall,
                "mrr": mrr,
                "hit": hit,
                "skipped": skipped,
                "retrieval_time_ms": benchmark_result.retrieval_time_ms,
            }

        # Process queries concurrently
        total_time = 0
        hits = 0
        completed = 0
        skipped_count = 0
        valid_recalls = []
        valid_mrrs = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_query, q): q for q in queries}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results["recall_at_k"].append(result["recall"])
                    results["mrr"].append(result["mrr"])
                    total_time += result["retrieval_time_ms"]
                    
                    # Track skipped queries separately
                    if result.get("skipped"):
                        skipped_count += 1
                    else:
                        valid_recalls.append(result["recall"])
                        valid_mrrs.append(result["mrr"])
                        if result["hit"]:
                            hits += 1
                except Exception as e:
                    # On error, append zeros
                    results["recall_at_k"].append(0.0)
                    results["mrr"].append(0.0)
                    print(f"\nError processing query: {e}")

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(queries))
                
                # Save intermediate results periodically
                if output_file and completed % save_interval == 0:
                    valid_count = completed - skipped_count
                    current_recall = sum(valid_recalls) / len(valid_recalls) if valid_recalls else 0
                    current_mrr = sum(valid_mrrs) / len(valid_mrrs) if valid_mrrs else 0
                    current_hit_rate = hits / valid_count if valid_count else 0
                    
                    intermediate_results = {
                        "status": "in_progress",
                        "completed": completed,
                        "total": len(queries),
                        "percent": round(completed / len(queries) * 100, 2),
                        "avg_recall_at_k": round(current_recall, 4),
                        "avg_mrr": round(current_mrr, 4),
                        "hit_rate": round(current_hit_rate, 4),
                        "skipped_queries": skipped_count,
                        "valid_queries": valid_count,
                        "estimated_final_recall": round(current_recall, 4),  # Current best estimate
                    }
                    try:
                        import json
                        with open(output_file, "w") as f:
                            json.dump(intermediate_results, f, indent=2)
                        # Also write a monitoring-friendly format
                        monitor_file = str(output_file).replace(".json", "_monitor.txt")
                        with open(monitor_file, "w") as f:
                            f.write(f"=== MultiHop-RAG Benchmark Progress ===\n")
                            f.write(f"Progress: {completed}/{len(queries)} ({completed/len(queries)*100:.1f}%)\n")
                            f.write(f"Current Recall@10: {current_recall*100:.2f}%\n")
                            f.write(f"Current Hit Rate: {current_hit_rate*100:.2f}%\n")
                            f.write(f"Current MRR: {current_mrr:.4f}\n")
                            f.write(f"Skipped: {skipped_count} queries (no evidence)\n")
                    except Exception:
                        pass  # Don't fail on save errors

        # Aggregate metrics (excluding skipped queries)
        valid_count = len(queries) - skipped_count
        results["avg_recall_at_k"] = sum(valid_recalls) / len(valid_recalls) if valid_recalls else 0
        results["avg_mrr"] = sum(valid_mrrs) / len(valid_mrrs) if valid_mrrs else 0
        results["hit_rate"] = hits / valid_count if valid_count else 0
        results["avg_retrieval_time_ms"] = total_time / len(queries) if queries else 0
        results["skipped_queries"] = skipped_count
        results["valid_queries"] = valid_count

        return results


class RAGBenchEvaluator:
    """
    Evaluator for the RAGBench benchmark.

    Implements TRACe evaluation metrics.
    """

    def __init__(self, adapter: UltimateRAGAdapter, data_dir: str):
        self.adapter = adapter
        self.data_dir = Path(data_dir)

    def load_dataset(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        Load RAGBench dataset.

        Note: RAGBench is typically loaded from HuggingFace:
            from datasets import load_dataset
            dataset = load_dataset("rungalileo/ragbench")
        """
        # Try local file first
        local_path = self.data_dir / f"{split}.json"
        if local_path.exists():
            with open(local_path) as f:
                return json.load(f)

        raise FileNotFoundError(
            f"Dataset not found at {local_path}. "
            "Load from HuggingFace: load_dataset('rungalileo/ragbench')"
        )

    def evaluate_trace_metrics(
        self,
        queries: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Evaluate using TRACe metrics.

        TRACe = Truthfulness, Relevance, Adherence, Completeness

        Args:
            queries: List of query dicts
            top_k: Number of chunks to retrieve

        Returns:
            TRACe metrics dict
        """
        results = {
            "utilization": [],  # How much of retrieved context is used
            "relevance": [],    # Are retrieved docs relevant
            "adherence": [],    # Does answer stick to context (no hallucination)
            "completeness": [], # Does answer cover all aspects
        }

        for q in queries:
            query_text = q.get("query", q.get("question", ""))
            ground_truth = q.get("answer", q.get("ground_truth", ""))

            # Full RAG pipeline
            benchmark_result = self.adapter.retrieve_and_generate(
                query_text,
                top_k=top_k,
                ground_truth=ground_truth
            )

            # Note: Full TRACe evaluation requires LLM-as-judge
            # This is a simplified version using text overlap
            if benchmark_result.generated_answer and ground_truth:
                answer_lower = benchmark_result.generated_answer.lower()
                truth_lower = ground_truth.lower()

                # Simplified completeness: word overlap
                truth_words = set(truth_lower.split())
                answer_words = set(answer_lower.split())
                overlap = len(truth_words & answer_words)
                completeness = overlap / len(truth_words) if truth_words else 0
                results["completeness"].append(completeness)

        # Aggregate
        return {
            "avg_completeness": sum(results["completeness"]) / len(results["completeness"]) if results["completeness"] else 0,
            "total_evaluated": len(queries),
        }


class CRAGEvaluator:
    """
    Evaluator for Meta's CRAG benchmark.

    Implements the scoring: correct (+1), missing (0), hallucination (-1)
    """

    def __init__(self, adapter: UltimateRAGAdapter, data_dir: str):
        self.adapter = adapter
        self.data_dir = Path(data_dir)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load CRAG dataset."""
        dataset_path = self.data_dir / "data"

        queries = []
        for json_file in dataset_path.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    queries.extend(data)
                else:
                    queries.append(data)

        return queries

    def score_answer(
        self,
        generated: str,
        ground_truth: str,
        acceptable_answers: Optional[List[str]] = None,
    ) -> int:
        """
        Score an answer using CRAG scoring.

        Returns:
            +1 for correct
            0 for missing/abstain
            -1 for incorrect/hallucination
        """
        generated = generated.strip().lower()
        ground_truth = ground_truth.strip().lower()

        # Check for abstention
        abstention_phrases = ["i don't know", "cannot answer", "no information", "unclear"]
        if any(phrase in generated for phrase in abstention_phrases):
            return 0  # Missing

        # Check for correct answer
        if ground_truth in generated:
            return 1  # Correct

        # Check acceptable alternatives
        if acceptable_answers:
            for alt in acceptable_answers:
                if alt.lower() in generated:
                    return 1

        # Otherwise, hallucination
        return -1

    def evaluate(
        self,
        queries: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Evaluate on CRAG benchmark.

        Args:
            queries: CRAG query dicts
            top_k: Number of chunks to retrieve

        Returns:
            CRAG metrics including accuracy and hallucination rate
        """
        results = {
            "correct": 0,
            "missing": 0,
            "hallucination": 0,
            "total": len(queries),
        }

        for q in queries:
            query_text = q.get("query", q.get("question", ""))
            ground_truth = q.get("answer", "")
            acceptable = q.get("acceptable_answers", [])

            # Generate answer
            benchmark_result = self.adapter.retrieve_and_generate(
                query_text,
                top_k=top_k,
                ground_truth=ground_truth,
            )

            # Score
            score = self.score_answer(
                benchmark_result.generated_answer or "",
                ground_truth,
                acceptable,
            )

            if score == 1:
                results["correct"] += 1
            elif score == 0:
                results["missing"] += 1
            else:
                results["hallucination"] += 1

        # Calculate rates
        total = results["total"]
        results["accuracy"] = results["correct"] / total if total else 0
        results["hallucination_rate"] = results["hallucination"] / total if total else 0
        results["crag_score"] = (results["correct"] - results["hallucination"]) / total if total else 0

        return results


def run_quick_test(api_url: str = "http://localhost:8000"):
    """Quick test to verify the adapter works."""
    print(f"Testing Ultimate RAG adapter at {api_url}")

    adapter = UltimateRAGAdapter(api_url=api_url)

    # Health check
    if not adapter.health_check():
        print("ERROR: Server not healthy or not reachable")
        return False
    print("Health check passed")

    # Test retrieval
    results, meta = adapter.retrieve("test query", top_k=3)
    print(f"Retrieval test: got {len(results)} results in {meta['retrieval_time_ms']:.1f}ms")
    print(f"Strategies used: {meta['strategies_used']}")

    return True


if __name__ == "__main__":
    import sys

    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    run_quick_test(api_url)
