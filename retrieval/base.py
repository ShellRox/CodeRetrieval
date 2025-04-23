"""
base.py

Base class for all retrieval methods. Defines the common interface and utility functions.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import pickle
import json


class CodeRetriever(ABC):
    """
    Abstract base class for code retrieval systems.

    All concrete retrieval implementations should inherit from this class
    and implement the required methods.
    """

    def __init__(self, index_dir: str = None):
        """
        Initialize the retriever.

        Args:
            index_dir: Directory to save/load index files
        """
        self.index_dir = index_dir
        self.documents = []
        self.is_indexed = False

        # Create index directory if it doesn't exist
        if index_dir and not os.path.exists(index_dir):
            os.makedirs(index_dir)

    @abstractmethod
    def index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build the search index from a list of documents.

        Args:
            documents: List of code documents to index
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search the index for the given query.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of retrieved documents with relevance scores
        """
        pass

    def save_index(self, filename: str = None) -> None:
        """
        Save the index to disk.

        Args:
            filename: Name of the file to save the index (default: retriever class name)
        """
        if not self.index_dir:
            raise ValueError("Index directory not specified")

        if not self.is_indexed:
            raise ValueError("No index to save. Call index() first.")

        if filename is None:
            filename = f"{self.__class__.__name__}.pkl"

        filepath = os.path.join(self.index_dir, filename)

        # Save the index using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.get_index_data(), f)

        # Save document metadata separately (in JSON format for easier inspection)
        metadata = [{
            'id': doc['id'],
            'func_name': doc.get('func_name', ''),
            'repo': doc.get('repo', ''),
            'path': doc.get('path', '')
        } for doc in self.documents]

        metadata_path = os.path.join(self.index_dir, f"{self.__class__.__name__}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    def load_index(self, filename: str = None) -> bool:
        """
        Load the index from disk.

        Args:
            filename: Name of the file to load the index from (default: retriever class name)

        Returns:
            True if index was loaded successfully, False otherwise
        """
        if not self.index_dir:
            raise ValueError("Index directory not specified")

        if filename is None:
            filename = f"{self.__class__.__name__}.pkl"

        filepath = os.path.join(self.index_dir, filename)

        if not os.path.exists(filepath):
            return False

        # Load the index
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)

        # Load document metadata
        metadata_path = os.path.join(self.index_dir, f"{self.__class__.__name__}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.documents = json.load(f)

        self.set_index_data(index_data)
        self.is_indexed = True
        return True

    @abstractmethod
    def get_index_data(self) -> Any:
        """
        Get the index data for serialization.

        Returns:
            Index data to be serialized
        """
        pass

    @abstractmethod
    def set_index_data(self, index_data: Any) -> None:
        """
        Set the index data after deserialization.

        Args:
            index_data: Index data from deserialization
        """
        pass

    def evaluate(self, queries: List[Dict[str, Any]], k: int = 10) -> Dict[str, float]:
        """
        Evaluate the retriever on a set of queries with ground truth.

        Args:
            queries: List of query dictionaries with 'text' and 'relevant_docs' fields
            k: Number of results to retrieve per query

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_indexed:
            raise ValueError("No index available. Call index() or load_index() first.")

        results = {
            'precision@1': 0.0,
            'precision@3': 0.0,
            'precision@5': 0.0,
            'precision@k': 0.0,
            'recall@k': 0.0,
            'mrr': 0.0,
            'map': 0.0,
        }

        if not queries:
            return results

        total_queries = len(queries)

        for query in queries:
            query_text = query['text']
            relevant_docs = set(query.get('relevant_docs', []))

            if not relevant_docs:
                total_queries -= 1
                continue

            # Get search results
            search_results = self.search(query_text, k)
            retrieved_docs = [result['id'] for result in search_results]

            # Calculate precision@k
            for n in [1, 3, 5, k]:
                if n > len(retrieved_docs):
                    continue
                precision_at_n = len(set(retrieved_docs[:n]) & relevant_docs) / n
                if n == 1:
                    results['precision@1'] += precision_at_n
                elif n == 3:
                    results['precision@3'] += precision_at_n
                elif n == 5:
                    results['precision@5'] += precision_at_n
                elif n == k:
                    results['precision@k'] += precision_at_n

            # Calculate recall@k
            recall = len(set(retrieved_docs) & relevant_docs) / len(relevant_docs)
            results['recall@k'] += recall

            # Calculate MRR (Mean Reciprocal Rank)
            mrr = 0.0
            for i, doc_id in enumerate(retrieved_docs):
                if doc_id in relevant_docs:
                    mrr = 1.0 / (i + 1)
                    break
            results['mrr'] += mrr

            # Calculate MAP (Mean Average Precision)
            avg_precision = 0.0
            relevant_found = 0
            for i, doc_id in enumerate(retrieved_docs):
                if doc_id in relevant_docs:
                    relevant_found += 1
                    precision_at_i = relevant_found / (i + 1)
                    avg_precision += precision_at_i

            if relevant_found > 0:
                avg_precision /= len(relevant_docs)
            results['map'] += avg_precision

        # Average the metrics
        if total_queries > 0:
            for metric in results:
                results[metric] /= total_queries

        return results