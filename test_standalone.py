#!/usr/bin/env python3
"""
Quick test to verify the standalone RAG benchmarking repo works.

Usage:
    python test_standalone.py
"""

import sys
import os

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all major components can be imported."""
    print("Testing imports...")
    
    # Core
    from ultimate_rag.core.node import KnowledgeNode, KnowledgeTree, TreeForest
    print("  ✓ Core (node, tree, forest)")
    
    # Retrieval
    from ultimate_rag.retrieval.retriever import UltimateRetriever, RetrievalConfig
    from ultimate_rag.retrieval.strategies import RetrievalStrategy
    print("  ✓ Retrieval (retriever, strategies)")
    
    # Reranking
    from ultimate_rag.retrieval.reranker import Reranker, EnsembleReranker
    print("  ✓ Reranking (reranker, ensemble)")
    
    # Graph
    from ultimate_rag.graph.graph import KnowledgeGraph
    from ultimate_rag.graph.entities import Entity
    print("  ✓ Graph (graph, entities)")
    
    # Ingestion
    from ultimate_rag.ingestion.processor import DocumentProcessor
    print("  ✓ Ingestion (processor)")
    
    # RAPTOR (knowledge_base)
    from knowledge_base.raptor import ClusterTreeBuilder, OpenAIEmbeddingModel
    print("  ✓ RAPTOR (cluster_tree_builder, embeddings)")
    
    # Adapters
    from adapters.ultimate_rag_adapter import UltimateRAGAdapter
    print("  ✓ Adapters (ultimate_rag_adapter)")
    
    print("\nAll imports successful! ✓")


def test_basic_functionality():
    """Test basic data structures work."""
    print("\nTesting basic functionality...")
    
    from ultimate_rag.core.node import KnowledgeNode, KnowledgeTree, TreeForest
    
    # Create a node
    node = KnowledgeNode(
        index="test-001",
        text="This is a test document about RAG systems.",
        layer=0
    )
    print(f"  ✓ Created node: {node.index}")
    
    # Create a tree
    tree = KnowledgeTree(tree_id="test-tree", name="Test Tree")
    tree.add_node(node)
    print(f"  ✓ Created tree with {len(tree.all_nodes)} node(s)")
    
    # Create a forest
    forest = TreeForest(forest_id="test-forest", name="Test Forest")
    forest.add_tree(tree)
    print(f"  ✓ Created forest with {len(forest.trees)} tree(s)")
    
    print("\nBasic functionality works! ✓")


def test_adapter():
    """Test the benchmark adapter can be instantiated."""
    print("\nTesting adapter...")
    
    from adapters.ultimate_rag_adapter import UltimateRAGAdapter
    
    # Create adapter (will fail to connect, that's OK)
    adapter = UltimateRAGAdapter(
        api_url="http://localhost:8000",
        retrieval_mode="standard"
    )
    print(f"  ✓ Created adapter with mode: {adapter.retrieval_mode}")
    
    print("\nAdapter works! ✓")


def main():
    print("=" * 60)
    print("RAG Benchmarking - Standalone Test")
    print("=" * 60)
    
    try:
        test_imports()
        test_basic_functionality()
        test_adapter()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Start server: cd ultimate_rag && python -m api.server")
        print("  3. Run benchmark: python scripts/run_multihop_eval.py --queries 10")
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        print("\nMake sure you've installed requirements:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
