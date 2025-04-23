"""
evaluation.py

Evaluation utilities for code retrieval systems.
Implements various metrics and evaluation methodologies.
"""

import os
import json
import time
import random
import traceback

import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def compute_precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    """
    Compute precision@k.

    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs
        k: Cutoff rank

    Returns:
        Precision@k value
    """
    # Ensure k is not larger than the number of retrieved documents
    k = min(k, len(retrieved_docs))

    if k == 0:
        return 0.0

    # Convert to sets for faster lookup
    relevant_set = set(relevant_docs)

    # Count relevant documents in the top-k results
    relevant_count = sum(1 for doc_id in retrieved_docs[:k] if doc_id in relevant_set)

    return relevant_count / k


def compute_recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    """
    Compute recall@k.

    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs
        k: Cutoff rank

    Returns:
        Recall@k value
    """
    if not relevant_docs:
        return 1.0  # Perfect recall if there are no relevant documents

    # Ensure k is not larger than the number of retrieved documents
    k = min(k, len(retrieved_docs))

    # Convert to sets for faster lookup
    relevant_set = set(relevant_docs)

    # Count relevant documents in the top-k results
    relevant_count = sum(1 for doc_id in retrieved_docs[:k] if doc_id in relevant_set)

    return relevant_count / len(relevant_set)


def compute_mean_reciprocal_rank(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs

    Returns:
        MRR value
    """
    # Convert to sets for faster lookup
    relevant_set = set(relevant_docs)

    # Find the rank of the first relevant document
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)

    return 0.0  # No relevant document found


def compute_mean_average_precision(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
    """
    Compute Mean Average Precision (MAP).

    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs

    Returns:
        MAP value
    """
    if not relevant_docs:
        return 1.0  # Perfect MAP if there are no relevant documents

    # Convert to sets for faster lookup
    relevant_set = set(relevant_docs)

    # Compute average precision
    relevant_count = 0
    sum_precision = 0.0

    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_set:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            sum_precision += precision_at_i

    if relevant_count == 0:
        return 0.0

    return sum_precision / len(relevant_set)


def compute_f1_score(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    """
    Compute F1 score at rank k.

    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs
        k: Cutoff rank

    Returns:
        F1 score
    """
    precision = compute_precision_at_k(retrieved_docs, relevant_docs, k)
    recall = compute_recall_at_k(retrieved_docs, relevant_docs, k)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def evaluate_retriever(
    retriever: Any,
    queries: List[Dict[str, Any]],
    k_values: List[int] = None
) -> Dict[str, Any]:
    """
    Evaluate a retriever on a set of queries with ground truth.

    Args:
        retriever: Retriever object with a search method
        queries: List of query dictionaries with 'text' and 'relevant_docs' fields
        k_values: List of k values for precision@k and recall@k (default: [1, 3, 5, 10])

    Returns:
        Dictionary with evaluation metrics
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    if not queries:
        return {}

    # Initialize metrics
    metrics = {
        'mrr': 0.0,
        'map': 0.0,
    }

    for k in k_values:
        metrics[f'precision@{k}'] = 0.0
        metrics[f'recall@{k}'] = 0.0
        metrics[f'f1@{k}'] = 0.0

    # Add per-category metrics
    category_metrics = defaultdict(lambda: {
        'count': 0,
        'mrr': 0.0,
        'map': 0.0,
    })

    for k in k_values:
        for category in set(query.get('category', 'unknown') for query in queries):
            category_metrics[category][f'precision@{k}'] = 0.0
            category_metrics[category][f'recall@{k}'] = 0.0
            category_metrics[category][f'f1@{k}'] = 0.0

    # Track query execution time
    total_time = 0.0

    # Process each query
    for query in queries:
        query_text = query['text']
        relevant_docs = query.get('relevant_docs', [])
        category = query.get('category', 'unknown')

        if not relevant_docs:
            continue

        # Perform search and measure time
        start_time = time.time()
        results = retriever.search(query_text, max(k_values))
        query_time = time.time() - start_time
        total_time += query_time

        # Extract document IDs
        retrieved_docs = [result['id'] for result in results]

        # Compute metrics
        mrr = compute_mean_reciprocal_rank(retrieved_docs, relevant_docs)
        map_score = compute_mean_average_precision(retrieved_docs, relevant_docs)

        # Update overall metrics
        metrics['mrr'] += mrr
        metrics['map'] += map_score

        # Update per-k metrics
        for k in k_values:
            precision = compute_precision_at_k(retrieved_docs, relevant_docs, k)
            recall = compute_recall_at_k(retrieved_docs, relevant_docs, k)
            f1 = compute_f1_score(retrieved_docs, relevant_docs, k)

            metrics[f'precision@{k}'] += precision
            metrics[f'recall@{k}'] += recall
            metrics[f'f1@{k}'] += f1

        # Update category metrics
        category_metrics[category]['count'] += 1
        category_metrics[category]['mrr'] += mrr
        category_metrics[category]['map'] += map_score

        for k in k_values:
            precision = compute_precision_at_k(retrieved_docs, relevant_docs, k)
            recall = compute_recall_at_k(retrieved_docs, relevant_docs, k)
            f1 = compute_f1_score(retrieved_docs, relevant_docs, k)

            category_metrics[category][f'precision@{k}'] += precision
            category_metrics[category][f'recall@{k}'] += recall
            category_metrics[category][f'f1@{k}'] += f1

    # Calculate averages for overall metrics
    total_queries = len([q for q in queries if q.get('relevant_docs', [])])

    if total_queries > 0:
        metrics['mrr'] /= total_queries
        metrics['map'] /= total_queries

        for k in k_values:
            metrics[f'precision@{k}'] /= total_queries
            metrics[f'recall@{k}'] /= total_queries
            metrics[f'f1@{k}'] /= total_queries

    # Calculate averages for per-category metrics
    for category, category_data in category_metrics.items():
        count = category_data['count']
        if count > 0:
            category_data['mrr'] /= count
            category_data['map'] /= count

            for k in k_values:
                category_data[f'precision@{k}'] /= count
                category_data[f'recall@{k}'] /= count
                category_data[f'f1@{k}'] /= count

    # Add execution time metrics
    metrics['total_time'] = total_time
    metrics['avg_time'] = total_time / total_queries if total_queries > 0 else 0

    # Add category metrics to the overall metrics
    metrics['per_category'] = dict(category_metrics)

    return metrics


def compare_retrievers(
    results: Dict[str, Dict[str, Any]],
    metrics: List[str] = None,
    categories: List[str] = None,
    output_dir: str = './results'
) -> None:
    """
    Compare multiple retrievers and generate visualizations.

    Args:
        results: Dictionary mapping retriever names to their evaluation results
        metrics: List of metrics to compare (default: ['precision@3', 'precision@10', 'mrr', 'map'])
        categories: List of categories to include (None for all)
        output_dir: Directory to save visualizations
    """
    if metrics is None:
        metrics = ['precision@3', 'precision@10', 'mrr', 'map']

    if not results:
        print("No results to compare")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get retriever names
    retriever_names = list(results.keys())

    # Get all categories if not specified
    if not categories:
        categories = set()
        for retriever_results in results.values():
            if 'per_category' in retriever_results:
                categories.update(retriever_results['per_category'].keys())
        categories = sorted(categories)

    # Compare overall metrics
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.8 / len(retriever_names)

    for i, retriever_name in enumerate(retriever_names):
        if retriever_name not in results:
            continue

        # Extract metric values
        values = [results[retriever_name].get(metric, 0) for metric in metrics]

        # Plot bar
        plt.bar(x + i * width - 0.4 + width / 2, values, width, label=retriever_name)

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Comparison of Retrieval Methods')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save figure
    plt.savefig(os.path.join(output_dir, 'overall_comparison.png'))
    plt.close()

    # Compare per-category metrics
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(categories))
        width = 0.8 / len(retriever_names)

        for i, retriever_name in enumerate(retriever_names):
            if retriever_name not in results or 'per_category' not in results[retriever_name]:
                continue

            # Extract metric values for each category
            values = []
            for category in categories:
                if category in results[retriever_name]['per_category']:
                    values.append(results[retriever_name]['per_category'][category].get(metric, 0))
                else:
                    values.append(0)

            # Plot bar
            plt.bar(x + i * width - 0.4 + width / 2, values, width, label=retriever_name)

        plt.xlabel('Categories')
        plt.ylabel('Score')
        plt.title(f'Comparison of {metric} Across Categories')
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save figure
        plt.savefig(os.path.join(output_dir, f'category_comparison_{metric}.png'))
        plt.close()

    # Generate summary report
    report = "# Code Retrieval Evaluation Report\n\n"

    # Overall metrics table
    report += "## Overall Metrics\n\n"
    report += "| Retriever |"
    for metric in metrics:
        report += f" {metric} |"
    report += "\n"

    report += "| --- |"
    for _ in metrics:
        report += " --- |"
    report += "\n"

    for retriever_name in retriever_names:
        if retriever_name not in results:
            continue

        report += f"| {retriever_name} |"
        for metric in metrics:
            report += f" {results[retriever_name].get(metric, 0):.4f} |"
        report += "\n"

    # Per-category metrics tables
    if categories:
        report += "\n## Per-Category Metrics\n\n"

        for metric in metrics:
            report += f"### {metric}\n\n"
            report += "| Retriever |"
            for category in categories:
                report += f" {category} |"
            report += "\n"

            report += "| --- |"
            for _ in categories:
                report += " --- |"
            report += "\n"

            for retriever_name in retriever_names:
                if retriever_name not in results or 'per_category' not in results[retriever_name]:
                    continue

                report += f"| {retriever_name} |"
                for category in categories:
                    if category in results[retriever_name]['per_category']:
                        report += f" {results[retriever_name]['per_category'][category].get(metric, 0):.4f} |"
                    else:
                        report += " N/A |"
                report += "\n"

            report += "\n"

    # Execution time comparison
    report += "\n## Execution Time\n\n"
    report += "| Retriever | Average Time (s) | Total Time (s) |\n"
    report += "| --- | --- | --- |\n"

    for retriever_name in retriever_names:
        if retriever_name not in results:
            continue

        avg_time = results[retriever_name].get('avg_time', 0)
        total_time = results[retriever_name].get('total_time', 0)
        report += f"| {retriever_name} | {avg_time:.4f} | {total_time:.4f} |\n"

    # Save report
    with open(os.path.join(output_dir, 'evaluation_report.md'), 'w') as f:
        f.write(report)

    print(f"Evaluation report and visualizations saved to {output_dir}")


def load_annotations(annotation_file: str) -> List[Dict[str, Any]]:
    """
    Load query annotations from a file.

    Args:
        annotation_file: Path to the annotation file

    Returns:
        List of query dictionaries
    """
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    return annotations


def create_annotation_template(
    queries: List[str],
    categories: List[str] = None,
    output_file: str = 'annotations/template.json'
) -> None:
    """
    Create an annotation template file for manual annotation.

    Args:
        queries: List of query strings
        categories: List of query categories
        output_file: Path to save the template
    """
    if categories is None:
        categories = ['basic', 'medium', 'complex']

    # Create template
    template = []
    for i, query in enumerate(queries):
        # Assign a category based on index (for even distribution)
        category = categories[i % len(categories)]

        template.append({
            'text': query,
            'relevant_docs': [],
            'category': category,
            'notes': 'Add document IDs of relevant code snippets'
        })

    # Save template
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"Annotation template created at {output_file}")


class RetrievalEvaluator:
    """
    Evaluator class for code retrieval systems.
    Manages evaluation of multiple retrievers and query sets.
    """

    def __init__(self, output_dir: str = './results'):
        """
        Initialize the evaluator.

        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        self.results = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def add_retriever(self, name: str, retriever: Any) -> None:
        """
        Register a retriever for evaluation.

        Args:
            name: Name of the retriever
            retriever: Retriever object with a search method
        """
        self.results[name] = {'retriever': retriever}

    def evaluate(
        self,
        queries: List[Dict[str, Any]],
        k_values: List[int] = None
    ) -> None:
        """
        Evaluate all registered retrievers.

        Args:
            queries: List of query dictionaries
            k_values: List of k values for evaluation (default: [1, 3, 5, 10])
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        for name, data in self.results.items():
            print(f"Evaluating {name}...")

            # Skip if no retriever is registered
            if 'retriever' not in data:
                continue

            # Evaluate retriever
            metrics = evaluate_retriever(data['retriever'], queries, k_values)

            # Store results
            self.results[name].update(metrics)

    def save_results(self) -> None:
        """
        Save evaluation results to disk.
        """
        # Clean results (remove retriever objects)
        clean_results = {}
        for name, data in self.results.items():
            clean_results[name] = {k: v for k, v in data.items() if k != 'retriever'}

        # Save as JSON
        with open(os.path.join(self.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(clean_results, f, indent=2)

    def generate_report(
        self,
        metrics: List[str] = None,
        categories: List[str] = None
    ) -> None:
        """
        Generate evaluation report and visualizations.

        Args:
            metrics: List of metrics to include (default: ['precision@3', 'precision@10', 'mrr', 'map'])
            categories: List of categories to include
        """
        if metrics is None:
            metrics = ['precision@3', 'precision@10', 'mrr', 'map']

        # Clean results
        clean_results = {}
        for name, data in self.results.items():
            clean_results[name] = {k: v for k, v in data.items() if k != 'retriever'}

        # Generate report and visualizations
        compare_retrievers(clean_results, metrics, categories, self.output_dir)


# Function to check for fine-tuned model weights
def get_codebert_model_path():
    """Check if fine-tuned model weights exist and return the appropriate path."""
    finetuned_path = os.path.join("models", "codebert-finetuned", "best_model")
    # Check if the safetensors file exists
    if os.path.exists(os.path.join(finetuned_path, "model.safetensors")):
        logger.info(f"Using fine-tuned CodeBERT model from {finetuned_path}")
        return finetuned_path
    else:
        logger.info("Fine-tuned model not found, using default CodeBERT")
        return "microsoft/codebert-base"


# Function for creating hybrid retrievers
def get_hybrid_results(query: str, retrievers_dict: dict, retriever_names: list, weights: list, k: int = 10) -> List[Dict[str, Any]]:
    """
    Get combined results from multiple retrievers with normalized scores.

    Args:
        query: Query string
        retrievers_dict: Dictionary of retriever objects
        retriever_names: List of retriever names to combine
        weights: List of weights for each retriever (must match retriever_names length)
        k: Number of results to retrieve per query

    Returns:
        List of combined search results
    """
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
                    else:
                        all_results[doc_id] = result.copy()
                        all_results[doc_id]['score'] = norm_score
            else:
                # If all scores are the same, just use the original score
                for result in results:
                    doc_id = result['id']
                    if doc_id in all_results:
                        all_results[doc_id]['score'] += result['score'] * weights[i]
                    else:
                        all_results[doc_id] = result.copy()
                        all_results[doc_id]['score'] = result['score'] * weights[i]

    # Sort by combined score
    sorted_results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)

    # Return top k
    return sorted_results[:k]


# Helper function to evaluate hybrid retrievers
def evaluate_hybrid_retriever(name, retriever_names, weights, queries, k):
    """
    Evaluate a hybrid retriever created by combining multiple retrievers.

    Args:
        name: Name of the hybrid retriever
        retriever_names: List of retriever names to combine
        weights: List of weights for each retriever
        queries: List of query dictionaries
        k: Number of results to retrieve per query

    Returns:
        Evaluation metrics and time
    """
    print(f"Evaluating {name} on {len(queries)} queries...")
    eval_start_time = time.time()

    # Initialize metrics
    metrics = {
        'precision@1': 0.0,
        'precision@3': 0.0,
        'precision@5': 0.0,
        'precision@10': 0.0,
        'recall@10': 0.0,
        'mrr': 0.0,
        'map': 0.0,
        'total_time': 0.0,
        'per_query_times': [],
        'per_query_positions': []
    }

    # Process each query
    for i, query in enumerate(queries):
        query_text = query['text']
        relevant_docs = query['relevant_docs']

        # Execute search with hybrid approach
        query_start_time = time.time()
        retrieved_results = get_hybrid_results(
            query_text,
            retrievers,
            retriever_names,
            weights,
            k
        )
        retrieved_docs = [result['id'] for result in retrieved_results]
        query_time = time.time() - query_start_time

        # Calculate position of relevant document
        position = -1
        for j, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                position = j + 1
                break

        # Record position
        metrics['per_query_positions'].append(position)

        # Calculate metrics
        precision_at_1 = compute_precision_at_k(retrieved_docs, relevant_docs, 1)
        precision_at_3 = compute_precision_at_k(retrieved_docs, relevant_docs, 3)
        precision_at_5 = compute_precision_at_k(retrieved_docs, relevant_docs, 5)
        precision_at_10 = compute_precision_at_k(retrieved_docs, relevant_docs, 10)
        recall_at_10 = compute_recall_at_k(retrieved_docs, relevant_docs, 10)
        mrr = compute_mean_reciprocal_rank(retrieved_docs, relevant_docs)
        map_score = compute_mean_average_precision(retrieved_docs, relevant_docs)

        # Update metrics
        metrics['precision@1'] += precision_at_1
        metrics['precision@3'] += precision_at_3
        metrics['precision@5'] += precision_at_5
        metrics['precision@10'] += precision_at_10
        metrics['recall@10'] += recall_at_10
        metrics['mrr'] += mrr
        metrics['map'] += map_score
        metrics['total_time'] += query_time
        metrics['per_query_times'].append(query_time)

        # Print progress
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(queries)} queries ({(i + 1) / len(queries) * 100:.1f}%)")

    # Calculate averages
    num_queries = len(queries)
    metrics['precision@1'] /= num_queries
    metrics['precision@3'] /= num_queries
    metrics['precision@5'] /= num_queries
    metrics['precision@10'] /= num_queries
    metrics['recall@10'] /= num_queries
    metrics['mrr'] /= num_queries
    metrics['map'] /= num_queries
    metrics['avg_time'] = metrics['total_time'] / num_queries

    # Calculate rank statistics
    positions = [pos for pos in metrics['per_query_positions'] if pos > 0]
    metrics['found_ratio'] = len(positions) / num_queries
    metrics['avg_position'] = sum(positions) / len(positions) if positions else -1

    # Calculate time statistics
    metrics['min_time'] = min(metrics['per_query_times'])
    metrics['max_time'] = max(metrics['per_query_times'])

    eval_time = time.time() - eval_start_time
    print(f"Evaluation completed in {eval_time:.2f} seconds")

    return metrics, eval_time


# Helper function to print metrics
def print_metrics(metrics, name):
    """Print evaluation metrics in a standardized format."""
    print(f"\nEvaluation Results for {name}:")
    print(f"Found ratio: {metrics['found_ratio'] * 100:.2f}%")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"MAP: {metrics['map']:.4f}")
    print(f"Precision@1: {metrics['precision@1']:.4f}")
    print(f"Precision@3: {metrics['precision@3']:.4f}")
    print(f"Precision@5: {metrics['precision@5']:.4f}")
    print(f"Precision@10: {metrics['precision@10']:.4f}")
    print(f"Recall@10: {metrics['recall@10']:.4f}")
    print(f"Average position (when found): {metrics['avg_position']:.2f}")
    print(f"Average query time: {metrics['avg_time'] * 1000:.2f} ms")
    print(f"Min query time: {metrics['min_time'] * 1000:.2f} ms")
    print(f"Max query time: {metrics['max_time'] * 1000:.2f} ms")


if __name__ == "__main__":
    import os
    import json
    import argparse
    import time
    import random
    from preprocessing import load_codesearchnet_data, create_code_document
    from retrieval.bm25 import BM25CodeRetriever
    from retrieval.ast_based import ASTCodeRetriever

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # VERY IMPORTANT FOR MAC OS

    try:
        from retrieval.codebert.codebert import CodeBERTRetriever
    except ImportError:
        logger.warning("CodeBERT retriever not available. Install necessary dependencies to use it.")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate code retrieval models")
    parser.add_argument("--data_dir", type=str, default="./data/python",
                        help="Directory containing CodeSearchNet data")
    parser.add_argument("--subset", type=str, default="valid",
                        choices=["train", "valid", "test"],
                        help="Data subset to use")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--tokenization", type=str, default="subtokens",
                        choices=["simple", "subtokens", "ast_based"],
                        help="Tokenization method for BM25")
    parser.add_argument("--bm25_variant", type=str, default="okapi",
                        choices=["okapi", "plus"],
                        help="BM25 variant to use")
    parser.add_argument("--retrievers", type=str, nargs="+", default=["bm25"],
                        choices=["bm25", "ast", "codebert", "bm25+ast", "codebert+bm25", "codebert+ast",
                                 "codebert+bm25+ast"],
                        help="Retrievers to evaluate")
    parser.add_argument("--num_queries", type=int, default=100,
                        help="Number of queries to evaluate (0 for all)")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of results to retrieve per query")
    args = parser.parse_args()

    print(f"Evaluating code retrieval on {args.subset} data...", flush=True)
    print(f"- Retrievers: {', '.join(args.retrievers)}", flush=True)
    if "bm25" in args.retrievers or "bm25+ast" in args.retrievers or "codebert+bm25" in args.retrievers or "codebert+bm25+ast" in args.retrievers:
        print(f"- BM25 Tokenization: {args.tokenization}", flush=True)
        print(f"- BM25 variant: {args.bm25_variant}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    print(f"Loading data from {args.data_dir}...", flush=True)
    data = load_codesearchnet_data(args.data_dir, args.subset)
    print(f"Loaded {len(data)} examples in {time.time() - start_time:.2f} seconds", flush=True)

    doc_start_time = time.time()
    print("Processing documents...", flush=True)
    documents = [create_code_document(item) for item in data]
    print(f"Processed {len(documents)} documents in {time.time() - doc_start_time:.2f} seconds", flush=True)
    doc_id_to_idx = {doc['id']: i for i, doc in enumerate(documents)}

    print("Creating queries from docstrings...", flush=True)
    query_start_time = time.time()
    queries = []
    # Now include 'code' as one of the possible categories.
    possible_categories = ['docstring', 'func_name', 'repo', 'code']
    for doc in documents:
        docstring = doc.get('docstring', '').strip()
        if not docstring:
            continue
        words = docstring.split()
        if len(words) < 4:
            continue
        first_sentence = docstring.split('.')[0].strip() + '.'
        category = random.choice(possible_categories)
        query = {
            'text': first_sentence,
            'relevant_docs': [doc['id']],
            'category': category,
            'doc_id': doc['id']
        }
        queries.append(query)
    print(f"Created {len(queries)} queries in {time.time() - query_start_time:.2f} seconds", flush=True)

    if args.num_queries > 0 and args.num_queries < len(queries):
        random.seed(42)
        queries = random.sample(queries, args.num_queries)
        print(f"Randomly sampled {args.num_queries} queries for evaluation", flush=True)

    with open(os.path.join(args.output_dir, 'evaluation_queries.json'), 'w') as f:
        json.dump(queries[:100], f, indent=2)

    retrievers = {}
    if "bm25" in args.retrievers or "bm25+ast" in args.retrievers or "codebert+bm25" in args.retrievers or "codebert+bm25+ast" in args.retrievers:
        print("Initializing BM25 retriever...", flush=True)
        bm25_retriever = BM25CodeRetriever(
            tokenization_method=args.tokenization,
            bm25_variant=args.bm25_variant
        )
        index_start_time = time.time()
        print("Building BM25 index...", flush=True)
        bm25_retriever.index(documents)
        print(f"Built BM25 index in {time.time() - index_start_time:.2f} seconds", flush=True)
        retrievers["bm25"] = bm25_retriever

    if "ast" in args.retrievers or "bm25+ast" in args.retrievers or "codebert+ast" in args.retrievers or "codebert+bm25+ast" in args.retrievers:
        print("Initializing AST-based retriever...", flush=True)
        ast_retriever = ASTCodeRetriever(
            use_token_features=True,
            token_weight=0.3
        )
        index_start_time = time.time()
        print("Building AST index...", flush=True)
        ast_retriever.index(documents)
        print(f"Built AST index in {time.time() - index_start_time:.2f} seconds", flush=True)
        retrievers["ast"] = ast_retriever

    if "codebert" in args.retrievers or "codebert+bm25" in args.retrievers or "codebert+ast" in args.retrievers or "codebert+bm25+ast" in args.retrievers:
        print("Initializing CodeBERT retriever...", flush=True)
        try:
            model_path = get_codebert_model_path()
            faiss_index_path = os.path.join("models", "codebert-finetuned", "faiss_index.pkl")
            index_dir = os.path.dirname(faiss_index_path)
            codebert_retriever = CodeBERTRetriever(
                model_path=model_path,
                use_faiss=True,
                pooling_strategy="mean",
                normalize_embeddings=True,
                index_dir=index_dir
            )
            index_loaded = False
            if os.path.exists(faiss_index_path):
                print(f"Found existing FAISS index at {faiss_index_path}. Attempting to load...", flush=True)
                try:
                    index_loaded = codebert_retriever.load_index(os.path.basename(faiss_index_path))
                    if index_loaded:
                        print("Successfully loaded existing FAISS index", flush=True)
                        if not hasattr(codebert_retriever, 'documents') or not codebert_retriever.documents:
                            print("Index loaded but documents missing. Adding documents...", flush=True)
                            codebert_retriever.documents = documents
                            print(f"Added {len(documents)} documents to the loaded index", flush=True)
                    else:
                        print("Failed to load existing index, will rebuild", flush=True)
                except Exception as e:
                    print(f"Error loading existing index: {e}", flush=True)
                    print("Will rebuild the index", flush=True)
                    index_loaded = False
            if not index_loaded:
                index_start_time = time.time()
                print("Building CodeBERT index...", flush=True)
                codebert_retriever.index(documents)
                print(f"Built CodeBERT index in {time.time() - index_start_time:.2f} seconds", flush=True)
                print(f"Saving FAISS index to {faiss_index_path}...", flush=True)
                codebert_retriever.save_index(os.path.basename(faiss_index_path))
                print("Index saved successfully", flush=True)
            retrievers["codebert"] = codebert_retriever
        except Exception as e:
            logger.error(f"Failed to initialize CodeBERT retriever: {e}")
            logger.error(traceback.format_exc())
            logger.error("Skipping CodeBERT-based retrievers")
            args.retrievers = [r for r in args.retrievers if "codebert" not in r]

    results = {}

    def evaluate_single_retriever(name, retriever, queries, k):
        print(f"Evaluating {name} on {len(queries)} queries...", flush=True)
        eval_start_time = time.time()
        metrics = {
            'precision@1': 0.0,
            'precision@3': 0.0,
            'precision@5': 0.0,
            'precision@10': 0.0,
            'recall@10': 0.0,
            'mrr': 0.0,
            'map': 0.0,
            'total_time': 0.0,
            'per_query_times': [],
            'per_query_positions': []
        }
        for i, query in enumerate(queries):
            query_text = query['text']
            relevant_docs = query['relevant_docs']
            query_start_time = time.time()
            search_results = retriever.search(query_text, k)
            query_time = time.time() - query_start_time
            retrieved_docs = [result['id'] for result in search_results]
            position = -1
            for j, doc_id in enumerate(retrieved_docs):
                if doc_id in relevant_docs:
                    position = j + 1
                    break
            metrics['per_query_positions'].append(position)
            precision_at_1 = compute_precision_at_k(retrieved_docs, relevant_docs, 1)
            precision_at_3 = compute_precision_at_k(retrieved_docs, relevant_docs, 3)
            precision_at_5 = compute_precision_at_k(retrieved_docs, relevant_docs, 5)
            precision_at_10 = compute_precision_at_k(retrieved_docs, relevant_docs, 10)
            recall_at_10 = compute_recall_at_k(retrieved_docs, relevant_docs, 10)
            mrr = compute_mean_reciprocal_rank(retrieved_docs, relevant_docs)
            map_score = compute_mean_average_precision(retrieved_docs, relevant_docs)
            metrics['precision@1'] += precision_at_1
            metrics['precision@3'] += precision_at_3
            metrics['precision@5'] += precision_at_5
            metrics['precision@10'] += precision_at_10
            metrics['recall@10'] += recall_at_10
            metrics['mrr'] += mrr
            metrics['map'] += map_score
            metrics['total_time'] += query_time
            metrics['per_query_times'].append(query_time)
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(queries)} queries ({(i + 1) / len(queries) * 100:.1f}%)", flush=True)
        num_queries = len(queries)
        metrics['precision@1'] /= num_queries
        metrics['precision@3'] /= num_queries
        metrics['precision@5'] /= num_queries
        metrics['precision@10'] /= num_queries
        metrics['recall@10'] /= num_queries
        metrics['mrr'] /= num_queries
        metrics['map'] /= num_queries
        metrics['avg_time'] = metrics['total_time'] / num_queries
        positions = [pos for pos in metrics['per_query_positions'] if pos > 0]
        metrics['found_ratio'] = len(positions) / num_queries
        metrics['avg_position'] = sum(positions) / len(positions) if positions else -1
        metrics['min_time'] = min(metrics['per_query_times'])
        metrics['max_time'] = max(metrics['per_query_times'])
        eval_time = time.time() - eval_start_time
        print(f"Evaluation completed in {eval_time:.2f} seconds", flush=True)
        return metrics, eval_time

    for name in [r for r in args.retrievers if "+" not in r]:
        if name in retrievers:
            metrics, eval_time = evaluate_single_retriever(name, retrievers[name], queries, args.k)
            results[name] = {
                'metrics': metrics,
                'evaluation_time': eval_time
            }
            print_metrics(metrics, name)

    hybrid_configurations = {
        "bm25+ast": (["bm25", "ast"], [0.7, 0.3]),
        "codebert+bm25": (["codebert", "bm25"], [0.6, 0.4]),
        "codebert+ast": (["codebert", "ast"], [0.7, 0.3]),
        "codebert+bm25+ast": (["codebert", "bm25", "ast"], [0.5, 0.4, 0.1])
    }
    for name in [r for r in args.retrievers if "+" in r]:
        if name in hybrid_configurations:
            retriever_names, weights = hybrid_configurations[name]
            if all(r in retrievers for r in retriever_names):
                metrics, eval_time = evaluate_hybrid_retriever(
                    name,
                    retriever_names,
                    weights,
                    queries,
                    args.k
                )
                results[name] = {
                    'metrics': metrics,
                    'evaluation_time': eval_time
                }
                print_metrics(metrics, name)
            else:
                logger.warning(f"Skipping {name} because one or more required retrievers are not available")

    detailed_results = {}
    for name, result in results.items():
        detailed_results[name] = {
            'metrics': result['metrics'].copy(),
            'evaluation_time': result['evaluation_time']
        }
        if 'per_query_times' in detailed_results[name]['metrics']:
            del detailed_results[name]['metrics']['per_query_times']
        if 'per_query_positions' in detailed_results[name]['metrics']:
            del detailed_results[name]['metrics']['per_query_positions']
    with open(os.path.join(args.output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(detailed_results, f, indent=2)
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    import matplotlib.pyplot as plt
    for name, result in results.items():
        metrics = result['metrics']
        vis_dir = os.path.join(args.output_dir, name)
        os.makedirs(vis_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        positions = [pos for pos in metrics['per_query_positions'] if pos > 0]
        plt.hist(positions, bins=range(1, args.k + 2), alpha=0.7, color='blue')
        plt.title(f'{name} - Distribution of Relevant Document Positions')
        plt.xlabel('Position in Search Results')
        plt.ylabel('Frequency')
        plt.xticks(range(1, args.k + 1))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(vis_dir, 'position_distribution.png'))
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.hist([t * 1000 for t in metrics['per_query_times']], bins=20, alpha=0.7, color='green')
        plt.title(f'{name} - Distribution of Query Execution Times')
        plt.xlabel('Execution Time (ms)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(vis_dir, 'time_distribution.png'))
        plt.close()

    # --- NEW: Category Comparison Plots (including precision@1 and recall@10) ---
    category_results = {}  # {retriever: {category: {mrr:[], map:[], precision@1:[], precision@3:[], precision@10:[], recall@10:[]}}}
    for retriever_name in results.keys():
        category_results[retriever_name] = {}
        for query in queries:
            cat = query.get('category', 'unknown')
            if cat not in category_results[retriever_name]:
                category_results[retriever_name][cat] = {
                    'mrr': [],
                    'map': [],
                    'precision@1': [],
                    'precision@3': [],
                    'precision@10': [],
                    'recall@10': []
                }
            # For hybrid retrievers, check hybrid_configurations
            if retriever_name in retrievers:
                search_results = retrievers[retriever_name].search(query['text'], args.k)
            elif retriever_name in hybrid_configurations:
                hybrid_names, hybrid_weights = hybrid_configurations[retriever_name]
                search_results = get_hybrid_results(query['text'], retrievers, hybrid_names, hybrid_weights, args.k)
            else:
                search_results = []
            retrieved_docs = [result['id'] for result in search_results]
            mrr = compute_mean_reciprocal_rank(retrieved_docs, query['relevant_docs'])
            map_score = compute_mean_average_precision(retrieved_docs, query['relevant_docs'])
            p1 = compute_precision_at_k(retrieved_docs, query['relevant_docs'], 1)
            p3 = compute_precision_at_k(retrieved_docs, query['relevant_docs'], 3)
            p10 = compute_precision_at_k(retrieved_docs, query['relevant_docs'], 10)
            r10 = compute_recall_at_k(retrieved_docs, query['relevant_docs'], 10)
            category_results[retriever_name][cat]['mrr'].append(mrr)
            category_results[retriever_name][cat]['map'].append(map_score)
            category_results[retriever_name][cat]['precision@1'].append(p1)
            category_results[retriever_name][cat]['precision@3'].append(p3)
            category_results[retriever_name][cat]['precision@10'].append(p10)
            category_results[retriever_name][cat]['recall@10'].append(r10)

    all_categories = set()
    for retriever_dict in category_results.values():
        all_categories.update(retriever_dict.keys())
    all_categories = sorted(list(all_categories))

    for metric in ['mrr', 'map', 'precision@1', 'precision@3', 'precision@10', 'recall@10']:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(all_categories))
        width = 0.8 / len(category_results)
        for i, retriever_name in enumerate(category_results):
            values = []
            for cat in all_categories:
                if cat in category_results[retriever_name]:
                    values.append(np.mean(category_results[retriever_name][cat][metric]))
                else:
                    values.append(0)
            plt.bar(x + i * width - 0.4 + width/2, values, width, label=retriever_name)
        plt.xlabel("Category")
        plt.ylabel(metric.upper())
        plt.title(f"Comparison of {metric.upper()} across Categories")
        plt.xticks(x, all_categories)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(args.output_dir, f"category_comparison_{metric}.png"))
        plt.close()

    # --- NEW: Generate per-model Markdown Reports with Category-wise Comparison ---
    for name, result in results.items():
        metrics = result['metrics']
        per_cat_table = ""
        if name in category_results:
            per_cat_table += "\n### Category-wise Comparison\n\n"
            per_cat_table += "| Category | MRR | MAP | Precision@1 | Precision@3 | Precision@10 | Recall@10 |\n"
            per_cat_table += "|----------|-----|-----|--------------|-------------|--------------|-----------|\n"
            for cat in all_categories:
                mrr_val = np.mean(category_results[name][cat]['mrr']) if cat in category_results[name] else 0
                map_val = np.mean(category_results[name][cat]['map']) if cat in category_results[name] else 0
                p1_val = np.mean(category_results[name][cat]['precision@1']) if cat in category_results[name] else 0
                p3_val = np.mean(category_results[name][cat]['precision@3']) if cat in category_results[name] else 0
                p10_val = np.mean(category_results[name][cat]['precision@10']) if cat in category_results[name] else 0
                r10_val = np.mean(category_results[name][cat]['recall@10']) if cat in category_results[name] else 0
                per_cat_table += f"| {cat} | {mrr_val:.4f} | {map_val:.4f} | {p1_val:.4f} | {p3_val:.4f} | {p10_val:.4f} | {r10_val:.4f} |\n"
        report = f"""# {name.upper()} Evaluation Report

## Configuration

- **Dataset:** {args.subset}
- **Number of Documents:** {len(documents)}
- **Number of Queries:** {len(queries)}
- **K Value:** {args.k}

## Overall Performance Metrics

| Metric | Value |
|--------|-------|
| MRR | {metrics['mrr']:.4f} |
| MAP | {metrics['map']:.4f} |
| Precision@1 | {metrics['precision@1']:.4f} |
| Precision@3 | {metrics['precision@3']:.4f} |
| Precision@5 | {metrics['precision@5']:.4f} |
| Precision@10 | {metrics['precision@10']:.4f} |
| Recall@10 | {metrics['recall@10']:.4f} |
| Found Ratio | {metrics['found_ratio'] * 100:.2f}% |
| Average Position (when found) | {metrics['avg_position']:.2f} |

## Efficiency Metrics

| Metric | Value |
|--------|-------|
| Average Query Time | {metrics['avg_time'] * 1000:.2f} ms |
| Min Query Time | {metrics['min_time'] * 1000:.2f} ms |
| Max Query Time | {metrics['max_time'] * 1000:.2f} ms |
| Total Evaluation Time | {result['evaluation_time']:.2f} seconds |
{per_cat_table}
"""
        with open(os.path.join(args.output_dir, f'{name}_report.md'), 'w') as f:
            f.write(report)

    # --- NEW: Overall Comparison Report ---
    if len(results) >= 1:
        comparison_report = "# Retrieval Methods Comparison\n\n## Overall Performance Metrics\n\n| Metric |"
        for rn in results.keys():
            comparison_report += f" {rn} |"
        comparison_report += "\n|--------|"
        for _ in results.keys():
            comparison_report += "-------|"
        comparison_report += "\n"
        for metric in ['mrr', 'map', 'precision@1', 'precision@3', 'precision@5', 'precision@10', 'recall@10',
                       'found_ratio', 'avg_position']:
            if metric == 'found_ratio':
                comparison_report += f"| Found Ratio |"
                for rn in results.keys():
                    comparison_report += f" {results[rn]['metrics'][metric] * 100:.2f}% |"
            elif metric == 'avg_position':
                comparison_report += f"| Avg Position |"
                for rn in results.keys():
                    comparison_report += f" {results[rn]['metrics'][metric]:.2f} |"
            else:
                comparison_report += f"| {metric.upper()} |"
                for rn in results.keys():
                    comparison_report += f" {results[rn]['metrics'][metric]:.4f} |"
            comparison_report += "\n"
        comparison_report += "\n## Efficiency Metrics\n\n| Metric |"
        for rn in results.keys():
            comparison_report += f" {rn} |"
        comparison_report += "\n|--------|"
        for _ in results.keys():
            comparison_report += "-------|"
        comparison_report += "\n"
        for metric in ['avg_time', 'min_time', 'max_time']:
            comparison_report += f"| {metric} |"
            for rn in results.keys():
                comparison_report += f" {results[rn]['metrics'][metric] * 1000:.2f} ms |"
            comparison_report += "\n"
        comparison_report += "| Evaluation Time |"
        for rn in results.keys():
            comparison_report += f" {results[rn]['evaluation_time']:.2f} s |"
        comparison_report += "\n"
        comparison_report += "\n## Overall Comparison Visualization\n\n![Comparison](comparison.png)\n\n## Conclusion\n\nThis report compares the performance of different code retrieval methods on the same dataset and queries.\n"
        with open(os.path.join(args.output_dir, 'comparison_report.md'), 'w') as f:
            f.write(comparison_report)

    print(f"\nReports saved to {args.output_dir}", flush=True)
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds total", flush=True)

