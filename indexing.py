"""
indexing.py

Utility script for building and managing search indexes.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
import time

from preprocessing import load_codesearchnet_data, create_code_document
from retrieval.bm25 import BM25CodeRetriever


def build_index(
    retriever_type: str,
    data_dir: str,
    index_dir: str,
    subset: str = 'train',
    limit: Optional[int] = None,
    **retriever_args
) -> None:
    """
    Build a search index using the specified retriever.

    Args:
        retriever_type: Type of retriever ('bm25', 'codebert', 'ast', 'hybrid')
        data_dir: Directory containing the data
        index_dir: Directory to save the index
        subset: Data subset ('train', 'valid', or 'test')
        limit: Maximum number of examples to index
        **retriever_args: Additional arguments for the retriever
    """
    print(f"Building {retriever_type} index from {subset} data...")

    # Create index directory if it doesn't exist
    os.makedirs(index_dir, exist_ok=True)

    # Load data
    start_time = time.time()
    print(f"Loading data from {data_dir}...")
    raw_data = load_codesearchnet_data(data_dir, subset, limit)
    print(f"Loaded {len(raw_data)} examples in {time.time() - start_time:.2f} seconds")

    # Process documents
    start_time = time.time()
    print("Processing documents...")
    documents = [create_code_document(item) for item in raw_data]
    print(f"Processed {len(documents)} documents in {time.time() - start_time:.2f} seconds")

    # Create and configure retriever
    if retriever_type.lower() == 'bm25':
        retriever = BM25CodeRetriever(index_dir=index_dir, **retriever_args)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    # Build index
    start_time = time.time()
    print("Building index...")
    retriever.index(documents)
    print(f"Built index in {time.time() - start_time:.2f} seconds")

    # Save index
    start_time = time.time()
    print("Saving index...")
    retriever.save_index()
    print(f"Saved index in {time.time() - start_time:.2f} seconds")

    # Save index metadata
    metadata = {
        'retriever_type': retriever_type,
        'data_dir': data_dir,
        'subset': subset,
        'num_documents': len(documents),
        'retriever_args': retriever_args,
        'index_time': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(index_dir, 'index_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Index built and saved to {index_dir}")


def load_retriever(
    retriever_type: str,
    index_dir: str,
    **retriever_args
) -> Any:
    """
    Load a retriever with its saved index.

    Args:
        retriever_type: Type of retriever ('bm25', 'codebert', 'ast', 'hybrid')
        index_dir: Directory containing the index
        **retriever_args: Additional arguments for the retriever

    Returns:
        Loaded retriever object
    """
    print(f"Loading {retriever_type} retriever from {index_dir}...")

    # Create retriever object
    if retriever_type.lower() == 'bm25':
        retriever = BM25CodeRetriever(index_dir=index_dir, **retriever_args)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    # Load index
    start_time = time.time()
    success = retriever.load_index()

    if success:
        print(f"Loaded index in {time.time() - start_time:.2f} seconds")
    else:
        print(f"Failed to load index from {index_dir}")

    return retriever


def perform_search(
    retriever: Any,
    query: str,
    k: int = 10,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Perform a search using the given retriever.

    Args:
        retriever: Retriever object
        query: Query string
        k: Number of results to retrieve
        verbose: Whether to print results

    Returns:
        List of search results
    """
    if verbose:
        print(f"Searching for: {query}")

    # Perform search
    start_time = time.time()
    results = retriever.search(query, k)
    search_time = time.time() - start_time

    if verbose:
        print(f"Found {len(results)} results in {search_time:.4f} seconds")

        # Print results
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
            print(f"ID: {result['id']}")
            print(f"File: {result.get('path', 'N/A')}")
            print(f"Function: {result.get('func_name', 'N/A')}")
            print("-" * 40)
            print(result.get('code', '')[:300] + ("..." if len(result.get('code', '')) > 300 else ''))
            print("-" * 40)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and manage code search indexes")

    # Common arguments
    parser.add_argument("--data_dir", type=str, default="./data/python",
                        help="Directory containing CodeSearchNet data")
    parser.add_argument("--index_dir", type=str, default="./indexes",
                        help="Directory to save/load indexes")

    # Subparsers for different operations
    subparsers = parser.add_subparsers(dest="operation", help="Operation to perform")

    # Index builder
    index_parser = subparsers.add_parser("build", help="Build a search index")
    index_parser.add_argument("--retriever", type=str, required=True,
                             choices=["bm25", "codebert", "ast", "hybrid"],
                             help="Type of retriever to use")
    index_parser.add_argument("--subset", type=str, default="train",
                             choices=["train", "valid", "test"],
                             help="Data subset to index")
    index_parser.add_argument("--limit", type=int, default=None,
                             help="Maximum number of examples to index")
    index_parser.add_argument("--tokenization", type=str, default="subtokens",
                             choices=["simple", "subtokens", "ast_based"],
                             help="Tokenization method for BM25")
    index_parser.add_argument("--bm25_variant", type=str, default="okapi",
                             choices=["okapi", "plus"],
                             help="BM25 variant to use")
    index_parser.add_argument("--k1", type=float, default=1.5,
                             help="BM25 parameter k1")
    index_parser.add_argument("--b", type=float, default=0.75,
                             help="BM25 parameter b")

    # Search tool
    search_parser = subparsers.add_parser("search", help="Search the index")
    search_parser.add_argument("--retriever", type=str, required=True,
                              choices=["bm25", "codebert", "ast", "hybrid"],
                              help="Type of retriever to use")
    search_parser.add_argument("--query", type=str, required=True,
                              help="Query string")
    search_parser.add_argument("--k", type=int, default=10,
                              help="Number of results to retrieve")

    args = parser.parse_args()

    # Perform operation
    if args.operation == "build":
        # Extract BM25-specific arguments
        retriever_args = {}
        if args.retriever == "bm25":
            retriever_args = {
                "tokenization_method": args.tokenization,
                "bm25_variant": args.bm25_variant,
                "k1": args.k1,
                "b": args.b
            }

        # Build index
        build_index(
            retriever_type=args.retriever,
            data_dir=args.data_dir,
            index_dir=args.index_dir,
            subset=args.subset,
            limit=args.limit,
            **retriever_args
        )

    elif args.operation == "search":
        # Load retriever
        retriever = load_retriever(
            retriever_type=args.retriever,
            index_dir=args.index_dir
        )

        # Perform search
        perform_search(
            retriever=retriever,
            query=args.query,
            k=args.k
        )

    else:
        parser.print_help()
