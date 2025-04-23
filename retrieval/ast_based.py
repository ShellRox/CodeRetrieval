"""
ast_based.py

Implementation of AST-based retrieval for code search.
Uses Abstract Syntax Tree (AST) parsing to capture code structure.

Key features:
- Parses Python code into ASTs to extract structural features
- Provides structure-aware search capabilities
- Can be combined with other retrieval methods for improved results
"""

import ast
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import pickle

from preprocessing import normalize_code, tokenize_code
from retrieval.base import CodeRetriever


class ASTFeatureExtractor:
    """
    Extracts features from Python code using AST parsing.
    """

    def __init__(self):
        """
        Initialize the AST feature extractor.
        """
        # Define node types of interest for feature extraction
        self.node_types = [
            ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom,
            ast.For, ast.While, ast.If, ast.Try, ast.ExceptHandler,
            ast.Dict, ast.List, ast.Set, ast.Call, ast.Assign,
            ast.Return, ast.Yield, ast.YieldFrom, ast.Raise,
            ast.Assert, ast.Lambda, ast.With
        ]

        # Map node types to string names
        self.node_type_names = {t: t.__name__ for t in self.node_types}

    def extract_features(self, code: str) -> Dict[str, Any]:
        """
        Extract AST-based features from code.

        Args:
            code: Python code string

        Returns:
            Dictionary of features
        """
        features = {
            'node_types': Counter(),
            'max_depth': 0,
            'avg_depth': 0.0,
            'num_nodes': 0,
            'control_flow_depth': 0,
            'function_calls': Counter(),
            'variable_names': Counter(),
            'import_names': [],
            'function_names': [],
            'class_names': []
        }

        try:
            # Parse code into AST
            tree = ast.parse(code)

            # Extract features
            self._count_node_types(tree, features)
            self._measure_tree_depth(tree, features)
            self._extract_names(tree, features)

            # Calculate average depth
            if features['num_nodes'] > 0:
                features['avg_depth'] = features['sum_depth'] / features['num_nodes']

            # Remove temporary counting feature
            if 'sum_depth' in features:
                del features['sum_depth']

        except SyntaxError:
            # Handle syntax errors gracefully
            pass

        return features

    def _count_node_types(self, tree: ast.AST, features: Dict[str, Any]) -> None:
        """
        Count occurrences of different AST node types.

        Args:
            tree: AST tree
            features: Features dictionary to update
        """
        # Count node types
        for node in ast.walk(tree):
            features['num_nodes'] += 1

            # Count specific node types
            for node_type in self.node_types:
                if isinstance(node, node_type):
                    features['node_types'][self.node_type_names[node_type]] += 1

    def _measure_tree_depth(self, tree: ast.AST, features: Dict[str, Any]) -> None:
        """
        Measure the depth and complexity of the AST.

        Args:
            tree: AST tree
            features: Features dictionary to update
        """
        features['sum_depth'] = 0

        def _get_depth(node, depth=0):
            # Update max depth
            if depth > features['max_depth']:
                features['max_depth'] = depth

            # Update sum of depths (for average calculation)
            features['sum_depth'] += depth

            # Measure control flow depth
            if isinstance(node, (ast.For, ast.While, ast.If, ast.Try)):
                if depth > features['control_flow_depth']:
                    features['control_flow_depth'] = depth

            # Recursively process children
            for child in ast.iter_child_nodes(node):
                _get_depth(child, depth + 1)

        _get_depth(tree)

    def _extract_names(self, tree: ast.AST, features: Dict[str, Any]) -> None:
        """
        Extract names of functions, classes, variables, and imports.

        Args:
            tree: AST tree
            features: Features dictionary to update
        """
        # Extract function and class names
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                features['function_names'].append(node.name)

            elif isinstance(node, ast.ClassDef):
                features['class_names'].append(node.name)

            elif isinstance(node, ast.Import):
                for name in node.names:
                    features['import_names'].append(name.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    prefix = node.module + "."
                else:
                    prefix = ""
                for name in node.names:
                    features['import_names'].append(prefix + name.name)

            elif isinstance(node, ast.Call) and hasattr(node, 'func'):
                # Extract function calls
                if hasattr(node.func, 'id'):
                    features['function_calls'][node.func.id] += 1
                elif hasattr(node.func, 'attr') and hasattr(node.func, 'value'):
                    # Handle method calls (e.g., obj.method())
                    features['function_calls'][node.func.attr] += 1

            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                # Extract variable names
                features['variable_names'][node.id] += 1


class ASTCodeRetriever(CodeRetriever):
    """
    AST-based code retrieval system.

    Uses AST parsing to extract structural features from code and perform
    structure-aware similarity matching.
    """

    def __init__(
            self,
            index_dir: str = None,
            use_token_features: bool = True,
            token_weight: float = 0.5
    ):
        """
        Initialize the AST-based code retriever.

        Args:
            index_dir: Directory to save/load index files
            use_token_features: Whether to combine AST features with token features
            token_weight: Weight for token features in the similarity calculation
        """
        super().__init__(index_dir)
        self.feature_extractor = ASTFeatureExtractor()
        self.use_token_features = use_token_features
        self.token_weight = token_weight

        # Index data
        self.ast_features = []
        self.token_features = []

        self.similarity_cache = {}

    def index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build the AST-based search index from a list of documents.

        Args:
            documents: List of code documents to index
        """
        # Store the documents
        self.documents = documents

        # Extract AST features for each document
        self.ast_features = []
        self.token_features = []

        for doc in documents:
            # Get code
            code = doc.get('normalized_code', doc.get('code', ''))

            # Extract AST features
            ast_features = self.feature_extractor.extract_features(code)
            self.ast_features.append(ast_features)

            # Extract token features if needed
            if self.use_token_features:
                tokens = tokenize_code(code, 'subtokens')
                self.token_features.append(Counter(tokens))

        self.is_indexed = True

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search the AST-based index for the given query.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of retrieved documents with relevance scores
        """
        if not self.is_indexed:
            raise ValueError("No index available. Call index() or load_index() first.")

        # Extract query features
        query_tokens = tokenize_code(query, 'subtokens')
        query_token_features = Counter(query_tokens)


        # Try to extract AST features if the query appears to be code
        # Otherwise, use a simplified approach based on keywords
        try:
            # Check if query might be code
            if "def " in query or "class " in query or "=" in query or ":" in query:
                query_ast_features = self.feature_extractor.extract_features(query)
            else:
                # Create a simpler feature representation for natural language queries
                query_ast_features = {
                    'node_types': Counter(),
                    'function_calls': Counter(),
                    'import_names': [],
                    'function_names': [],
                    'class_names': []
                }

                # Extract potential code elements from the query
                words = query.split()
                for word in words:
                    word = word.strip(".,():;[]{}").lower()
                    # Check for code-related terms
                    if word in ['function', 'def', 'method', 'fn', 'func']:
                        query_ast_features['node_types']['FunctionDef'] += 2  # Higher weight
                    elif word in ['class', 'interface', 'struct']:
                        query_ast_features['node_types']['ClassDef'] += 2
                    elif word in ['import', 'include', 'require', 'using']:
                        query_ast_features['node_types']['Import'] += 2
                    elif word in ['for', 'loop', 'iterate', 'foreach']:
                        query_ast_features['node_types']['For'] += 2
                    elif word in ['while', 'until']:
                        query_ast_features['node_types']['While'] += 2
                    elif word in ['if', 'else', 'elif', 'condition', 'conditional']:
                        query_ast_features['node_types']['If'] += 2
                    elif word in ['try', 'except', 'catch', 'finally', 'error', 'exception']:
                        query_ast_features['node_types']['Try'] += 2
                    elif len(word) > 2:  # Only consider words with length > 2
                        # Check if the word looks like a camelCase or snake_case identifier
                        if '_' in word or (any(c.islower() for c in word) and any(c.isupper() for c in word)):
                            query_ast_features['function_names'].append(word)
        except:
            # Fallback for query parsing errors
            query_ast_features = {
                'node_types': Counter(),
                'function_calls': Counter(),
                'import_names': [],
                'function_names': [],
                'class_names': []
            }

        # Calculate similarity scores
        # Check cache for this query
        if query in self.similarity_cache:
            scores = self.similarity_cache[query]
        else:
            scores = []
            for i, doc_ast_features in enumerate(self.ast_features):
                # Calculate AST feature similarity
                ast_similarity = self._calculate_ast_similarity(query_ast_features, doc_ast_features)

                # Calculate token feature similarity if enabled
                token_similarity = 0.0
                if self.use_token_features:
                    token_similarity = self._calculate_token_similarity(query_token_features, self.token_features[i])

                # Combine similarities
                if self.use_token_features:
                    similarity = (1 - self.token_weight) * ast_similarity + self.token_weight * token_similarity
                else:
                    similarity = ast_similarity

                scores.append(similarity)

            # Cache the results
            self.similarity_cache[query] = scores

        # Get top-k documents
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

    def _calculate_ast_similarity(
            self,
            query_features: Dict[str, Any],
            doc_features: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between query and document AST features.

        Args:
            query_features: Query AST features
            doc_features: Document AST features

        Returns:
            Similarity score
        """
        similarity = 0.0

        # Name matching (functions, classes, imports) - give higher weight
        # as direct name matches are strong indicators
        name_sim = 0.0
        name_count = 0

        for feature in ['function_names', 'class_names', 'import_names']:
            if query_features.get(feature) and doc_features.get(feature):
                query_names = set(query_features[feature])
                doc_names = set(doc_features[feature])

                if query_names and doc_names:
                    overlap = len(query_names.intersection(doc_names))
                    name_sim += overlap / max(len(query_names), 1)
                    name_count += 1

        if name_count > 0:
            similarity += 0.6 * (name_sim / name_count)  # Increased from 0.5

        # Node type similarity - less weight as these can be more generic
        if query_features.get('node_types') and doc_features.get('node_types'):
            node_type_sim = self._calculate_counter_similarity(
                query_features['node_types'], doc_features['node_types'])
            similarity += 0.2 * node_type_sim  # Decreased from 0.3

        # Function call similarity
        if query_features.get('function_calls') and doc_features.get('function_calls'):
            func_call_sim = self._calculate_counter_similarity(
                query_features['function_calls'], doc_features['function_calls'])
            similarity += 0.2 * func_call_sim  # Same weight

        return similarity

    def _calculate_token_similarity(self, query_tokens: Counter, doc_tokens: Counter) -> float:
        """
        Calculate similarity between query and document token features.

        Args:
            query_tokens: Query token counter
            doc_tokens: Document token counter

        Returns:
            Similarity score
        """
        return self._calculate_counter_similarity(query_tokens, doc_tokens)

    def _calculate_counter_similarity(self, counter1: Counter, counter2: Counter) -> float:
        """
        Calculate similarity between two counters using cosine similarity.

        Args:
            counter1: First counter
            counter2: Second counter

        Returns:
            Similarity score
        """
        # Get all keys
        all_keys = set(counter1.keys()).union(counter2.keys())

        # Calculate dot product
        dot_product = sum(counter1.get(k, 0) * counter2.get(k, 0) for k in all_keys)

        # Calculate magnitudes
        magnitude1 = sum(counter1.get(k, 0) ** 2 for k in all_keys) ** 0.5
        magnitude2 = sum(counter2.get(k, 0) ** 2 for k in all_keys) ** 0.5

        # Calculate cosine similarity
        if magnitude1 > 0 and magnitude2 > 0:
            return dot_product / (magnitude1 * magnitude2)
        else:
            return 0.0

    def get_index_data(self) -> Dict[str, Any]:
        """
        Get the index data for serialization.

        Returns:
            Dictionary with AST features and related data
        """
        return {
            'ast_features': self.ast_features,
            'token_features': self.token_features,
            'use_token_features': self.use_token_features,
            'token_weight': self.token_weight
        }

    def set_index_data(self, index_data: Dict[str, Any]) -> None:
        """
        Set the index data after deserialization.

        Args:
            index_data: Dictionary with AST features and related data
        """
        self.ast_features = index_data['ast_features']

        if 'token_features' in index_data:
            self.token_features = index_data['token_features']

        if 'use_token_features' in index_data:
            self.use_token_features = index_data['use_token_features']

        if 'token_weight' in index_data:
            self.token_weight = index_data['token_weight']