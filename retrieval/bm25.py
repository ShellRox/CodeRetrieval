"""
bm25.py

Implementation of BM25 retrieval model for code search.
Uses the rank_bm25 library for efficient BM25 implementation.

BM25 is a bag-of-words retrieval function that ranks documents based on the
query terms appearing in each document, regardless of their proximity within the document.
It is a probabilistic retrieval model that extends TF-IDF with term frequency normalization.

Key components:
- k1: Controls term frequency saturation (default: 1.5)
- b: Controls document length normalization (default: 0.75)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from rank_bm25 import BM25Okapi, BM25Plus
import os
import pickle

from preprocessing import tokenize_code, normalize_code
from retrieval.base import CodeRetriever


class BM25CodeRetriever(CodeRetriever):
    """
    BM25-based code retrieval system.

    Uses BM25 algorithm to rank code documents based on query relevance.
    Supports different tokenization methods and BM25 variants.
    """

    def __init__(
            self,
            index_dir: str = None,
            tokenization_method: str = 'subtokens',
            bm25_variant: str = 'okapi',
            k1: float = 1.5,
            b: float = 0.75
    ):
        """
        Initialize the BM25 code retriever.

        Args:
            index_dir: Directory to save/load index files
            tokenization_method: Method to tokenize code ('simple', 'subtokens', 'ast_based')
            bm25_variant: BM25 variant to use ('okapi' or 'plus')
            k1: BM25 parameter controlling term frequency saturation
            b: BM25 parameter controlling document length normalization
        """
        super().__init__(index_dir)
        self.tokenization_method = tokenization_method
        self.bm25_variant = bm25_variant
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.doc_tokens = []
        self.id_to_index = {}  # Maps document IDs to their indices in self.documents

    def index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build the BM25 index from a list of documents.

        Args:
            documents: List of code documents to index
        """
        # Store the documents
        self.documents = documents

        # Create a mapping from document ID to index
        self.id_to_index = {doc['id']: idx for idx, doc in enumerate(documents)}

        # Extract and tokenize code from each document
        self.doc_tokens = []
        for doc in documents:
            if self.tokenization_method in doc.get('tokens', {}):
                # Use pre-computed tokens if available
                tokens = doc['tokens'][self.tokenization_method]
            else:
                # Otherwise, tokenize the code
                code = doc.get('normalized_code', doc.get('code', ''))
                tokens = tokenize_code(code, self.tokenization_method)

            self.doc_tokens.append(tokens)

        # Initialize the BM25 model
        if self.bm25_variant.lower() == 'plus':
            self.bm25 = BM25Plus(self.doc_tokens, k1=self.k1, b=self.b)
        else:  # Default to Okapi
            self.bm25 = BM25Okapi(self.doc_tokens, k1=self.k1, b=self.b)

        self.is_indexed = True

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search the BM25 index for the given query.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of retrieved documents with relevance scores
        """
        if not self.is_indexed:
            raise ValueError("No index available. Call index() or load_index() first.")

        # Tokenize the query
        query_tokens = tokenize_code(query, self.tokenization_method)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Sort document indices by descending score
        top_indices = np.argsort(scores)[::-1][:k]

        # Prepare results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                doc = self.documents[idx]
                results.append({
                    'id': doc['id'],
                    'score': float(scores[idx]),
                    'code': doc.get('code', ''),
                    'func_name': doc.get('func_name', ''),
                    'repo': doc.get('repo', ''),
                    'path': doc.get('path', ''),
                    'docstring': doc.get('docstring', '')
                })

        return results

    def get_index_data(self) -> Dict[str, Any]:
        """
        Get the index data for serialization.

        Returns:
            Dictionary with BM25 model and related data
        """
        return {
            'bm25': self.bm25,
            'doc_tokens': self.doc_tokens,
            'id_to_index': self.id_to_index,
            'tokenization_method': self.tokenization_method,
            'bm25_variant': self.bm25_variant,
            'k1': self.k1,
            'b': self.b
        }

    def set_index_data(self, index_data: Dict[str, Any]) -> None:
        """
        Set the index data after deserialization.

        Args:
            index_data: Dictionary with BM25 model and related data
        """
        self.bm25 = index_data['bm25']
        self.doc_tokens = index_data['doc_tokens']
        self.id_to_index = index_data['id_to_index']
        self.tokenization_method = index_data['tokenization_method']
        self.bm25_variant = index_data['bm25_variant']
        self.k1 = index_data['k1']
        self.b = index_data['b']

    def optimize_parameters(
            self,
            queries: List[Dict[str, Any]],
            k1_values: List[float] = None,
            b_values: List[float] = None,
            k: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize BM25 parameters (k1 and b) using a grid search.

        Args:
            queries: List of query dictionaries with 'text' and 'relevant_docs' fields
            k1_values: List of k1 values to try (default: [0.5, 1.0, 1.5, 2.0, 2.5])
            b_values: List of b values to try (default: [0.0, 0.25, 0.5, 0.75, 1.0])
            k: Number of results to retrieve per query

        Returns:
            Dictionary with best parameters and evaluation results
        """
        if not self.is_indexed:
            raise ValueError("No index available. Call index() or load_index() first.")

        # Default parameter values to try
        if k1_values is None:
            k1_values = [0.5, 1.0, 1.5, 2.0, 2.5]
        if b_values is None:
            b_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        # Save original parameters
        original_k1 = self.k1
        original_b = self.b
        original_bm25 = self.bm25

        best_map = 0.0
        best_k1 = original_k1
        best_b = original_b
        best_results = None

        # Grid search over parameters
        for k1 in k1_values:
            for b in b_values:
                # Create a new BM25 model with current parameters
                if self.bm25_variant.lower() == 'plus':
                    self.bm25 = BM25Plus(self.doc_tokens, k1=k1, b=b)
                else:  # Default to Okapi
                    self.bm25 = BM25Okapi(self.doc_tokens, k1=k1, b=b)

                # Evaluate the model
                results = self.evaluate(queries, k)

                # Update best parameters if MAP improves
                if results['map'] > best_map:
                    best_map = results['map']
                    best_k1 = k1
                    best_b = b
                    best_results = results

        # Restore original parameters
        self.k1 = original_k1
        self.b = original_b

        # Return the best parameters and results
        optimization_results = {
            'best_k1': best_k1,
            'best_b': best_b,
            'best_map': best_map,
            'best_results': best_results
        }

        # Optionally update the model with best parameters
        self.k1 = best_k1
        self.b = best_b
        if self.bm25_variant.lower() == 'plus':
            self.bm25 = BM25Plus(self.doc_tokens, k1=best_k1, b=best_b)
        else:
            self.bm25 = BM25Okapi(self.doc_tokens, k1=best_k1, b=best_b)

        return optimization_results