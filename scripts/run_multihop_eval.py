#!/usr/bin/env python3
"""
Run MultiHop-RAG Evaluation on Ultimate RAG.

This script:
1. Loads the MultiHop-RAG corpus
2. Ingests documents into Ultimate RAG
3. Runs retrieval evaluation
4. Reports metrics (Recall@K, MRR, Hit Rate)

Usage:
    python run_multihop_eval.py --api-url http://localhost:8000 --top-k 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.ultimate_rag_adapter import UltimateRAGAdapter, MultiHopRAGEvaluator


def progress_bar(current: int, total: int, width: int = 40, prefix: str = ""):
    """Simple progress bar with ETA."""
    import sys
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    msg = f"\r{prefix}[{bar}] {current}/{total} ({percent*100:.1f}%)"
    sys.stdout.write(msg)
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Run MultiHop-RAG evaluation")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Ultimate RAG API URL")
    parser.add_argument("--data-dir", default=None, help="MultiHop-RAG data directory")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--max-queries", type=int, default=None, help="Max queries to evaluate (for testing)")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip corpus ingestion")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent workers (default: 10)")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds (default: 120)")
    parser.add_argument("--output", default="multihop_results.json", help="Output file for results")
    
    # RAPTOR hierarchy building options
    parser.add_argument(
        "--build-hierarchy", action="store_true",
        help="Build full RAPTOR tree hierarchy with clustering and summarization (RECOMMENDED for best performance)"
    )
    parser.add_argument(
        "--hierarchy-layers", type=int, default=5,
        help="Max number of tree layers when building hierarchy (default: 5)"
    )
    parser.add_argument(
        "--hierarchy-target-top", type=int, default=50,
        help="Target size for top layer when building hierarchy (default: 50)"
    )
    args = parser.parse_args()

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent / "multihop_rag"

    print(f"=== MultiHop-RAG Evaluation ===")
    print(f"API URL: {args.api_url}")
    print(f"Data directory: {data_dir}")
    print(f"Top-K: {args.top_k}")
    print(f"Concurrency: {args.concurrency}")
    if args.build_hierarchy:
        print(f"RAPTOR Hierarchy: ENABLED (layers={args.hierarchy_layers}, target_top={args.hierarchy_target_top})")
    else:
        print(f"RAPTOR Hierarchy: disabled (flat ingestion)")
    print("")

    # Initialize adapter and evaluator
    # Standard mode is optimal: 71.67% recall with cross-encoder reranking
    adapter = UltimateRAGAdapter(
        api_url=args.api_url,
        default_top_k=args.top_k,
        retrieval_mode="standard",
        timeout=args.timeout,
    )

    # Health check
    print("Checking API health...")
    if not adapter.health_check():
        print("ERROR: Ultimate RAG API not available!")
        print("Make sure the server is running at", args.api_url)
        sys.exit(1)
    print("API is healthy!")
    print("")

    evaluator = MultiHopRAGEvaluator(adapter, str(data_dir))

    # Always try to load persisted trees first
    tree_loaded = False
    try:
        available = adapter.list_available_trees()
        local_trees = available.get("local_trees", [])
        if local_trees:
            print(f"Found persisted trees: {local_trees}")
            print("Loading persisted data...")
            for tree_name in local_trees:
                load_result = adapter.load_tree(tree_name)
                if load_result.get("success"):
                    print(f"  Loaded: {tree_name} ({load_result.get('node_count', '?')} nodes)")
                    tree_loaded = True
                else:
                    print(f"  Failed to load: {tree_name}")
            print("")
    except Exception as e:
        print(f"Could not check for persisted trees: {e}")

    # Ingest if no tree loaded and not skipping
    if not args.skip_ingest and not tree_loaded:
        print("Loading corpus...")
        try:
            corpus = evaluator.load_corpus()
            print(f"Found {len(corpus)} documents in corpus")

            print("Ingesting corpus into Ultimate RAG...")
            if args.build_hierarchy:
                print("*** Building full RAPTOR hierarchy (this may take 10-30 minutes) ***")
            else:
                print("(Using batch embedding - this may take a few minutes)")
            start_time = time.time()

            def ingest_progress(current, total):
                progress_bar(current, total, prefix="Ingest: ")

            ingest_results = adapter.ingest_benchmark_corpus(
                corpus,
                progress_callback=ingest_progress,
                build_hierarchy=args.build_hierarchy,
                hierarchy_num_layers=args.hierarchy_layers,
                hierarchy_target_top_nodes=args.hierarchy_target_top,
            )
            print("")  # New line after progress bar

            elapsed = time.time() - start_time
            print(f"Ingestion complete in {elapsed:.1f}s")
            print(f"  Successful: {ingest_results['successful']}/{ingest_results['total']}")
            print(f"  Chunks created: {ingest_results['total_chunks']}")
            if args.build_hierarchy:
                print(f"  Total nodes: {ingest_results.get('total_nodes', 'N/A')}")
                print(f"  Tree layers: {ingest_results.get('num_layers', 0)}")
                layer_dist = ingest_results.get('layer_distribution', {})
                if layer_dist:
                    for layer in sorted(layer_dist.keys()):
                        print(f"    Layer {layer}: {layer_dist[layer]} nodes")
            print(f"  Entities found: {ingest_results['total_entities']}")
            if ingest_results.get('persisted'):
                print(f"  Data persisted to: {ingest_results.get('persist_path', 'disk')}")
            if ingest_results['errors']:
                print(f"  Errors: {len(ingest_results['errors'])}")
            print("")

        except FileNotFoundError as e:
            print(f"WARNING: Could not load corpus: {e}")
            print("Continuing with evaluation (assuming data already ingested)...")
            print("")

    # Load queries
    print("Loading evaluation queries...")
    try:
        queries = evaluator.load_dataset()
        print(f"Found {len(queries)} queries")
    except FileNotFoundError as e:
        print(f"ERROR: Could not load queries: {e}")
        print("Make sure you've downloaded the MultiHop-RAG dataset:")
        print("  cd rag_benchmarking/multihop_rag")
        print("  git clone https://github.com/yixuantt/MultiHop-RAG.git .")
        sys.exit(1)

    # Limit queries if specified
    if args.max_queries:
        queries = queries[:args.max_queries]
        print(f"Limited to {len(queries)} queries for evaluation")

    # Run evaluation
    print("")
    print(f"Running retrieval evaluation with {args.concurrency} concurrent workers...")
    start_time = time.time()

    def eval_progress(current, total):
        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0
        progress_bar(current, total, prefix=f"Eval ({rate:.1f}/s, ETA {eta:.0f}s): ")

    results = evaluator.evaluate_retrieval(
        queries,
        top_k=args.top_k,
        max_workers=args.concurrency,
        progress_callback=eval_progress,
        output_file=args.output,  # Save intermediate results
        save_interval=50,  # Save every 50 queries for monitoring
    )
    print("")  # New line after progress bar

    elapsed = time.time() - start_time
    print(f"Evaluation complete in {elapsed:.1f}s")
    print("")

    # Print results
    print("=== Results ===")
    print(f"Total queries: {results['total_queries']}")
    print(f"Recall@{args.top_k}: {results['avg_recall_at_k']:.4f}")
    print(f"MRR: {results['avg_mrr']:.4f}")
    print(f"Hit Rate: {results['hit_rate']:.4f}")
    print(f"Avg retrieval time: {results['avg_retrieval_time_ms']:.1f}ms")

    # Save results
    output_path = Path(args.output)
    results["config"] = {
        "api_url": args.api_url,
        "top_k": args.top_k,
        "data_dir": str(data_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
