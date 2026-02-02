#!/usr/bin/env python3
"""
HotpotQA Benchmark Evaluation Script for OpenRAG.

HotpotQA is a multi-hop QA benchmark with 113K queries requiring
reasoning across multiple Wikipedia documents.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.ultimate_rag_adapter import UltimateRAGAdapter


class HotpotQAEvaluator:
    """Evaluator for HotpotQA benchmark."""
    
    def __init__(self, adapter: UltimateRAGAdapter, dataset_path: str):
        self.adapter = adapter
        self.dataset_path = dataset_path
        self.queries = []
        self.corpus_docs = []
        
    def load_dataset(self, limit: int = None, offset: int = 0) -> int:
        """Load HotpotQA dataset."""
        print(f"Loading HotpotQA from {self.dataset_path}...")
        
        with open(self.dataset_path) as f:
            data = json.load(f)
        
        # Apply offset and limit
        if offset > 0:
            data = data[offset:]
        if limit:
            data = data[:limit]
            
        self.queries = data
        print(f"Loaded {len(self.queries)} queries")
        
        # Extract corpus from context (each query has its own context)
        # In fullwiki setting, we need to build a combined corpus
        doc_set = {}
        for item in self.queries:
            for i, title in enumerate(item['context']['title']):
                sentences = item['context']['sentences'][i]
                doc_text = ' '.join(sentences)
                doc_id = title
                if doc_id not in doc_set:
                    doc_set[doc_id] = {
                        'title': title,
                        'body': doc_text,
                        'id': doc_id
                    }
        
        self.corpus_docs = list(doc_set.values())
        print(f"Extracted {len(self.corpus_docs)} unique documents from contexts")
        
        return len(self.queries)
    
    def ingest_corpus(self, batch_size: int = 100, build_hierarchy: bool = True) -> bool:
        """Ingest HotpotQA corpus into the RAG system."""
        print(f"\nIngesting {len(self.corpus_docs)} documents...")
        
        # Prepare documents
        documents = []
        for doc in self.corpus_docs:
            content = f"{doc['title']}\n\n{doc['body']}"
            documents.append({
                'content': content,
                'metadata': {
                    'title': doc['title'],
                    'source': 'hotpotqa'
                }
            })
        
        # Batch ingest
        total_nodes = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            try:
                result = self.adapter.ingest_batch(
                    batch,
                    tree_name="default",
                    build_hierarchy=build_hierarchy
                )
                nodes = result.get('total_nodes_created', 0)
                total_nodes += nodes
                print(f"  Batch {i//batch_size + 1}: {i+len(batch)}/{len(documents)} docs, {total_nodes} nodes")
            except Exception as e:
                print(f"  Error in batch {i//batch_size + 1}: {e}")
        
        print(f"Ingestion complete: {total_nodes} nodes created")
        return total_nodes > 0
    
    def evaluate_single(self, item: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
        """Evaluate a single query."""
        query = item['question']
        
        # Get supporting facts (gold evidence)
        gold_titles = set(item['supporting_facts']['title'])
        
        try:
            # Retrieve
            result = self.adapter.retrieve(query, top_k=top_k)
            
            # Handle tuple return
            if isinstance(result, tuple):
                retrieved_chunks, metadata = result
            else:
                retrieved_chunks = result
                metadata = {}
            
            # Check which gold documents were retrieved
            retrieved_titles = set()
            for chunk in retrieved_chunks:
                # Handle both dict and RetrievalResult objects
                if hasattr(chunk, 'text'):
                    text = chunk.text
                elif isinstance(chunk, dict):
                    text = chunk.get('text', '') or chunk.get('content', '')
                else:
                    text = str(chunk)
                
                # Check if any gold title appears in the chunk
                for title in gold_titles:
                    if title.lower() in text.lower():
                        retrieved_titles.add(title)
            
            # Calculate recall
            if len(gold_titles) > 0:
                recall = len(retrieved_titles & gold_titles) / len(gold_titles)
            else:
                recall = 0.0
            
            return {
                'id': item['id'],
                'question': query,
                'answer': item['answer'],
                'type': item['type'],
                'level': item['level'],
                'gold_titles': list(gold_titles),
                'retrieved_titles': list(retrieved_titles),
                'recall': recall,
                'num_retrieved': len(retrieved_chunks),
                'success': True
            }
            
        except Exception as e:
            return {
                'id': item['id'],
                'question': query,
                'error': str(e),
                'recall': 0.0,
                'success': False
            }
    
    def evaluate_retrieval(
        self,
        top_k: int = 10,
        max_workers: int = 1,
        output_file: str = None,
        monitor_file: str = None
    ) -> Dict[str, float]:
        """Evaluate retrieval performance on HotpotQA."""
        print(f"\nEvaluating {len(self.queries)} queries (top_k={top_k}, workers={max_workers})...")
        
        results = []
        start_time = time.time()
        
        if max_workers == 1:
            # Serial execution with progress
            for i, item in enumerate(self.queries):
                result = self.evaluate_single(item, top_k)
                results.append(result)
                
                # Progress update
                if (i + 1) % 10 == 0 or i == len(self.queries) - 1:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (len(self.queries) - i - 1) / rate if rate > 0 else 0
                    
                    # Calculate running metrics
                    valid = [r for r in results if r['success']]
                    avg_recall = sum(r['recall'] for r in valid) / len(valid) if valid else 0
                    
                    print(f"  [{i+1}/{len(self.queries)}] Recall: {avg_recall*100:.2f}%, "
                          f"Rate: {rate:.1f}/s, ETA: {eta:.0f}s")
                    
                    # Save intermediate results
                    if output_file and (i + 1) % 50 == 0:
                        self._save_intermediate(results, output_file, monitor_file)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.evaluate_single, item, top_k): item
                    for item in self.queries
                }
                
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        valid = [r for r in results if r['success']]
                        avg_recall = sum(r['recall'] for r in valid) / len(valid) if valid else 0
                        print(f"  [{i+1}/{len(self.queries)}] Recall: {avg_recall*100:.2f}%")
        
        # Calculate final metrics
        valid_results = [r for r in results if r['success']]
        
        metrics = {
            'total_queries': len(self.queries),
            'successful_queries': len(valid_results),
            'failed_queries': len(results) - len(valid_results),
            'recall@10': sum(r['recall'] for r in valid_results) / len(valid_results) if valid_results else 0,
        }
        
        # Breakdown by type
        for qtype in ['comparison', 'bridge']:
            type_results = [r for r in valid_results if r.get('type') == qtype]
            if type_results:
                metrics[f'recall@10_{qtype}'] = sum(r['recall'] for r in type_results) / len(type_results)
        
        # Breakdown by level
        for level in ['easy', 'medium', 'hard']:
            level_results = [r for r in valid_results if r.get('level') == level]
            if level_results:
                metrics[f'recall@10_{level}'] = sum(r['recall'] for r in level_results) / len(level_results)
        
        # Save final results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump({
                    'metrics': metrics,
                    'results': results
                }, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return metrics
    
    def _save_intermediate(self, results: List[Dict], output_file: str, monitor_file: str = None):
        """Save intermediate results for monitoring."""
        valid = [r for r in results if r['success']]
        metrics = {
            'queries_completed': len(results),
            'recall@10': sum(r['recall'] for r in valid) / len(valid) if valid else 0,
        }
        
        with open(output_file, 'w') as f:
            json.dump({'metrics': metrics, 'results': results}, f)
        
        if monitor_file:
            with open(monitor_file, 'w') as f:
                f.write(f"HotpotQA Progress: {len(results)} queries\n")
                f.write(f"Recall@10: {metrics['recall@10']*100:.2f}%\n")


def main():
    parser = argparse.ArgumentParser(description='Run HotpotQA evaluation')
    parser.add_argument('--api-url', default='http://localhost:8000', help='RAG API URL')
    parser.add_argument('--dataset', default='hotpotqa/validation.json', help='Dataset path')
    parser.add_argument('--limit', type=int, default=None, help='Limit queries')
    parser.add_argument('--offset', type=int, default=0, help='Query offset')
    parser.add_argument('--top-k', type=int, default=10, help='Top-K retrieval')
    parser.add_argument('--concurrency', type=int, default=1, help='Parallel workers')
    parser.add_argument('--output', default='hotpotqa_results.json', help='Output file')
    parser.add_argument('--skip-ingest', action='store_true', help='Skip ingestion')
    parser.add_argument('--build-hierarchy', action='store_true', help='Build RAPTOR hierarchy')
    parser.add_argument('--retrieval-mode', default='standard', help='Retrieval mode')
    
    args = parser.parse_args()
    
    # Initialize adapter
    adapter = UltimateRAGAdapter(
        api_url=args.api_url,
        retrieval_mode=args.retrieval_mode
    )
    
    # Initialize evaluator
    evaluator = HotpotQAEvaluator(adapter, args.dataset)
    
    # Load dataset
    evaluator.load_dataset(limit=args.limit, offset=args.offset)
    
    # Ingest if needed
    if not args.skip_ingest:
        evaluator.ingest_corpus(build_hierarchy=args.build_hierarchy)
    
    # Evaluate
    metrics = evaluator.evaluate_retrieval(
        top_k=args.top_k,
        max_workers=args.concurrency,
        output_file=args.output
    )
    
    # Print results
    print("\n" + "="*50)
    print("HOTPOTQA RESULTS")
    print("="*50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value*100:.2f}%")
        else:
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
