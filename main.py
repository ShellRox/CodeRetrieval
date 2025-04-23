"""
main.py

Script to evaluate different retrieval models on custom queries of varying difficulty.
"""

import os
import time
import json
import numpy as np
from typing import List, Dict, Any
from tabulate import tabulate

# Import retrieval models
from preprocessing import load_codesearchnet_data, create_code_document
from retrieval.bm25 import BM25CodeRetriever
from retrieval.ast_based import ASTCodeRetriever
from retrieval.codebert.codebert import CodeBERTRetriever

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # VERY IMPORTANT FOR MAC OS


# Function to get CodeBERT model path
def get_codebert_model_path():
    """Check if fine-tuned model weights exist and return the appropriate path."""
    finetuned_path = os.path.join("models", "codebert-finetuned", "best_model")
    if os.path.exists(finetuned_path):
        print(f"Using fine-tuned CodeBERT model from {finetuned_path}")
        return finetuned_path
    else:
        print("Fine-tuned model not found, using default CodeBERT")
        return "microsoft/codebert-base"


# Function for hybrid retrieval
def get_hybrid_results(query: str, retrievers_dict: dict, retriever_names: list, weights: list, k: int = 10) -> List[
    Dict[str, Any]]:
    """Get combined results from multiple retrievers with normalized scores."""
    if len(retriever_names) != len(weights):
        raise ValueError("Number of retrievers and weights must match")

    # Get results from each retriever
    all_results = {}
    for i, name in enumerate(retriever_names):
        if name not in retrievers_dict:
            continue

        # Get results
        results = retrievers_dict[name].search(query, k * 2)  # Get more results to ensure good coverage

        # Extract scores
        scores = []
        for result in results:
            scores.append(result['score'])

        # Normalize scores to 0-1 range if there are scores
        if scores:
            max_score = max(scores)
            min_score = min(scores)
            score_range = max_score - min_score

            # Avoid division by zero
            if score_range > 0:
                # Add normalized scores with weight
                for result in results:
                    doc_id = result['id']
                    norm_score = (result['score'] - min_score) / score_range * weights[i]

                    if doc_id in all_results:
                        all_results[doc_id]['score'] += norm_score
                        all_results[doc_id]['retrievers'].append(name)
                    else:
                        all_results[doc_id] = result.copy()
                        all_results[doc_id]['score'] = norm_score
                        all_results[doc_id]['retrievers'] = [name]
            else:
                # If all scores are the same, just use the original score
                for result in results:
                    doc_id = result['id']
                    if doc_id in all_results:
                        all_results[doc_id]['score'] += result['score'] * weights[i]
                        all_results[doc_id]['retrievers'].append(name)
                    else:
                        all_results[doc_id] = result.copy()
                        all_results[doc_id]['score'] = result['score'] * weights[i]
                        all_results[doc_id]['retrievers'] = [name]

    # Sort by combined score
    sorted_results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)

    # Return top k
    return sorted_results[:k]


# Define custom queries of varying difficulty
queries = {
    "easy": [
        {"text": "How to read a file in Python", "description": "Basic file I/O operation"},
        {"text": "Function to sort a list", "description": "Common sorting operation"},
        {"text": "How to create a dictionary", "description": "Basic data structure creation"},
        {"text": "Function to calculate average", "description": "Simple arithmetic operation"},
        {"text": "How to connect to database", "description": "Basic database connection"}
    ],
    "medium": [
        {"text": "Implement a binary search algorithm", "description": "Common search algorithm"},
        {"text": "How to handle authentication in web apps", "description": "Web security concept"},
        {"text": "Efficient way to find duplicates in an array", "description": "Algorithmic efficiency"},
        {"text": "Implement error handling for API requests", "description": "Error management"},
        {"text": "How to use decorators in Python", "description": "Advanced Python feature"}
    ],
    "complex": [
        {"text": "Implement a thread-safe singleton pattern",
         "description": "Advanced design pattern with concurrency"},
        {"text": "How to optimize database queries for large datasets", "description": "Performance optimization"},
        {"text": "Implement a custom caching mechanism", "description": "Advanced system design"},
        {"text": "Create a function to detect and prevent SQL injection", "description": "Security implementation"},
        {"text": "How to implement vector similarity search", "description": "Advanced algorithm implementation"}
    ]
}


def main():
    # Configuration
    data_dir = "./data/python"
    subset = "test"
    output_dir = "./custom_eval_results"
    k = 10

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    data = load_codesearchnet_data(data_dir, subset)
    print(f"Loaded {len(data)} examples")

    print("Processing documents...")
    documents = [create_code_document(item) for item in data]
    print(f"Processed {len(documents)} documents")

    # Initialize retrievers
    retrievers = {}

    # Initialize BM25
    print("Initializing BM25 retriever...")
    bm25_retriever = BM25CodeRetriever(
        tokenization_method="subtokens",
        bm25_variant="okapi"
    )
    print("Building BM25 index...")
    bm25_retriever.index(documents)
    retrievers["bm25"] = bm25_retriever

    # Initialize AST-based retriever
    print("Initializing AST-based retriever...")
    ast_retriever = ASTCodeRetriever(
        use_token_features=True,
        token_weight=0.3
    )
    print("Building AST index...")
    ast_retriever.index(documents)
    retrievers["ast"] = ast_retriever

    # Initialize CodeBERT
    print("Initializing CodeBERT retriever...")
    model_path = get_codebert_model_path()
    codebert_retriever = CodeBERTRetriever(
        model_path=model_path,
        use_faiss=True,
        pooling_strategy="mean",
        normalize_embeddings=True
    )

    # Try to load existing FAISS index
    faiss_index_path = os.path.join("models", "codebert-finetuned", "faiss_index.pkl")
    index_dir = os.path.dirname(faiss_index_path)
    codebert_retriever.index_dir = index_dir

    if os.path.exists(faiss_index_path):
        print(f"Loading existing FAISS index from {faiss_index_path}...")
        try:
            index_loaded = codebert_retriever.load_index(os.path.basename(faiss_index_path))
            if index_loaded:
                print("Successfully loaded existing FAISS index")
                if not hasattr(codebert_retriever, 'documents') or not codebert_retriever.documents:
                    print("Index loaded but documents missing. Adding documents...")
                    codebert_retriever.documents = documents
                    print(f"Added {len(documents)} documents to the loaded index")
        except Exception as e:
            print(f"Error loading index: {e}")
            print("Building CodeBERT index...")
            codebert_retriever.index(documents)
    else:
        print("No existing index found. Building CodeBERT index...")
        codebert_retriever.index(documents)

    retrievers["codebert"] = codebert_retriever

    # Define hybrid configurations
    hybrid_configs = {
        "bm25+ast": (["bm25", "ast"], [0.7, 0.3]),
        "codebert+bm25": (["codebert", "bm25"], [0.6, 0.4]),
        "codebert+ast": (["codebert", "ast"], [0.7, 0.3]),
        "codebert+bm25+ast": (["codebert", "bm25", "ast"], [0.5, 0.4, 0.1])
    }

    # Function to evaluate a query
    def evaluate_query(query, difficulty):
        results = {}

        # Evaluate single retrievers
        for name in ["bm25", "ast", "codebert"]:
            retriever = retrievers[name]
            start_time = time.time()
            search_results = retriever.search(query["text"], k)
            elapsed_time = time.time() - start_time

            results[name] = {
                "results": search_results[:3],  # Store top 3 results for display
                "time": elapsed_time * 1000  # Convert to ms
            }

        # Evaluate hybrid retrievers
        for name, (retriever_names, weights) in hybrid_configs.items():
            start_time = time.time()
            search_results = get_hybrid_results(query["text"], retrievers, retriever_names, weights, k)
            elapsed_time = time.time() - start_time

            results[name] = {
                "results": search_results[:3],  # Store top 3 results for display
                "time": elapsed_time * 1000  # Convert to ms
            }

        return results

    # Evaluate all queries
    all_results = {}

    for difficulty, query_list in queries.items():
        print(f"\nEvaluating {difficulty} queries...")
        difficulty_results = {}

        for i, query in enumerate(query_list):
            print(f"  Query {i + 1}/{len(query_list)}: {query['text']}")
            query_results = evaluate_query(query, difficulty)
            difficulty_results[query["text"]] = query_results

            # Print top result for each retriever
            print("  Top results:")
            headers = ["Retriever", "Top Function", "Score", "Query Time (ms)"]
            table_data = []

            for retriever_name, result in query_results.items():
                if result["results"]:
                    top_result = result["results"][0]
                    table_data.append([
                        retriever_name,
                        top_result.get("func_name", "N/A"),
                        f"{top_result['score']:.4f}",
                        f"{result['time']:.2f}"
                    ])
                else:
                    table_data.append([retriever_name, "No results", "N/A", f"{result['time']:.2f}"])

            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print()

        all_results[difficulty] = difficulty_results

    # Save results
    with open(os.path.join(output_dir, "custom_eval_results.json"), "w") as f:
        # Need to convert some objects to serializable format
        serializable_results = {}
        for difficulty, difficulty_results in all_results.items():
            serializable_results[difficulty] = {}
            for query_text, query_results in difficulty_results.items():
                serializable_results[difficulty][query_text] = {}
                for retriever_name, result in query_results.items():
                    serializable_results[difficulty][query_text][retriever_name] = {
                        "results": [
                            {
                                "id": r.get("id", ""),
                                "score": float(r.get("score", 0)),
                                "func_name": r.get("func_name", ""),
                                "path": r.get("path", ""),
                                "code": r.get("code", ""),
                                "retrievers": r.get("retrievers", [])
                            } for r in result["results"]
                        ],
                        "time": float(result["time"])
                    }

        json.dump(serializable_results, f, indent=2)

    print(f"\nEvaluation complete. Results saved to {os.path.join(output_dir, 'custom_eval_results.json')}")

    # Generate summary statistics
    summary = {
        "avg_time_ms": {},
    }

    for retriever_name in ["bm25", "ast", "codebert", "bm25+ast", "codebert+bm25", "codebert+ast", "codebert+bm25+ast"]:
        times = []
        for difficulty, difficulty_results in all_results.items():
            for query_text, query_results in difficulty_results.items():
                if retriever_name in query_results:
                    times.append(query_results[retriever_name]["time"])

        if times:
            summary["avg_time_ms"][retriever_name] = sum(times) / len(times)

    print("\nPerformance Summary:")
    print("Average Query Time (ms):")
    for retriever_name, avg_time in summary["avg_time_ms"].items():
        print(f"  {retriever_name}: {avg_time:.2f} ms")

    # Provide instructions for manual evaluation
    print("\nManual Evaluation Instructions:")
    print("1. Open the results file at:", os.path.join(output_dir, "custom_eval_results.json"))
    print("2. For each query, review the top results from each retriever.")
    print("3. Determine which result is most relevant to the query.")
    print("4. Compare how each retriever performs across different query difficulties.")


if __name__ == "__main__":
    main()