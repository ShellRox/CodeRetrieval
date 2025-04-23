"""
codebert.py

Implementation of CodeBERT-based retrieval for code search.
Uses the pre-trained or fine-tuned CodeBERT model to generate embeddings for code and queries.

Key features:
- Uses CodeBERT to create semantic embeddings of code
- Supports efficient similarity search with FAISS
- Can be combined with other retrieval methods for improved results
"""

import numpy as np
import os
import pickle
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import RobertaTokenizer, RobertaModel
import faiss
import logging

from preprocessing import normalize_code, tokenize_code
from retrieval.base import CodeRetriever

import json

# Configure logger
logger = logging.getLogger(__name__)


class CodeBERTRetriever(CodeRetriever):
    """
    CodeBERT-based code retrieval system.

    Uses the pre-trained or fine-tuned CodeBERT model to generate embeddings for code and queries,
    then performs similarity search to retrieve relevant code snippets.
    """

    def __init__(
        self,
        index_dir: str = None,
        model_path: str = "microsoft/codebert-base",
        use_faiss: bool = True,
        max_length: int = 512,
        pooling_strategy: str = "mean",
        normalize_embeddings: bool = True
    ):
        """
        Initialize the CodeBERT-based retriever.

        Args:
            index_dir: Directory to save/load index files
            model_path: Path to the pre-trained or fine-tuned model
            use_faiss: Whether to use FAISS for efficient similarity search
            max_length: Maximum sequence length for tokenization
            pooling_strategy: Strategy for pooling token embeddings ("mean", "max", or "cls")
            normalize_embeddings: Whether to normalize embeddings for cosine similarity
        """
        super().__init__(index_dir)
        self.model_path = model_path  # Fixed: using model_path instead of model_name
        self.use_faiss = use_faiss
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.normalize_embeddings = normalize_embeddings

        # Load model and tokenizer if available
        self.tokenizer = None
        self.model = None
        self.device = None

        # Index data
        self.embeddings = None
        self.faiss_index = None

        self.documents = []

    def _load_model(self) -> None:
        """
        Load the CodeBERT model and tokenizer.
        """
        try:
            logger.info(f"Loading tokenizer from {self.model_path}")
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path)

            logger.info(f"Loading model from {self.model_path}")
            self.model = RobertaModel.from_pretrained(self.model_path)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            self.model.to(self.device)
            self.model.eval()
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model and tokenizer: {str(e)}")
            raise

    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using CodeBERT.

        Args:
            text: Text to encode

        Returns:
            Embedding vector
        """
        # Load model if not loaded
        if self.model is None or self.tokenizer is None:
            self._load_model()

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]

        # Pool embeddings based on strategy
        if self.pooling_strategy == "cls":
            # Use [CLS] token embedding
            embedding = hidden_states[:, 0, :].cpu().numpy()
        elif self.pooling_strategy == "max":
            # Max pooling
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states = hidden_states * attention_mask_expanded
            embedding = torch.max(hidden_states, dim=1)[0].cpu().numpy()
        else:  # Default to mean pooling
            # Mean pooling
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, 1)
            sum_mask = torch.sum(attention_mask_expanded, 1)
            embedding = (sum_embeddings / sum_mask.clamp(min=1e-9)).cpu().numpy()

        return embedding[0]  # Return single embedding

    def index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build the CodeBERT index from a list of documents.

        Args:
            documents: List of code documents to index
        """
        # Store the documents
        self.documents = documents

        # Load model if not loaded
        if self.model is None or self.tokenizer is None:
            self._load_model()

        # Encode each document
        logger.info(f"Encoding {len(documents)} documents with CodeBERT...")
        embedding_dim = 768  # CodeBERT base has 768 dimensions
        self.embeddings = np.zeros((len(documents), embedding_dim))

        try:
            for i, doc in enumerate(documents):
                # Get code
                code = doc.get('normalized_code', doc.get('code', ''))

                # Encode code
                embedding = self._encode_text(code)
                self.embeddings[i] = embedding

                if (i + 1) % 100 == 0:
                    logger.info(f"Encoded {i + 1}/{len(documents)} documents")

            # Create FAISS index if enabled
            if self.use_faiss:
                logger.info("Building FAISS index...")
                self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])

                # Normalize embeddings for cosine similarity if requested
                if self.normalize_embeddings:
                    normalized_embeddings = self.embeddings.copy()
                    faiss.normalize_L2(normalized_embeddings)
                    self.faiss_index.add(normalized_embeddings)
                else:
                    self.faiss_index.add(self.embeddings)

            self.is_indexed = True
            logger.info("Indexing completed successfully")
        except Exception as e:
            logger.error(f"Error during indexing: {str(e)}")
            raise

    def save_index(self, filename: str = None) -> None:
        """
        Save the index to disk by saving embeddings and full document data.

        Args:
            filename: Name of the file to save the index data (default: retriever class name)
        """
        if not self.index_dir:
            raise ValueError("Index directory not specified")

        if not self.is_indexed:
            raise ValueError("No index to save. Call index() first.")

        if filename is None:
            filename = f"{self.__class__.__name__}.pkl"

        filepath = os.path.join(self.index_dir, filename)

        # Save all data needed for retrieval
        save_data = {
            'embeddings': self.embeddings,
            'documents': self.documents,  # Save the complete documents
            'model_path': self.model_path,
            'use_faiss': self.use_faiss,
            'max_length': self.max_length,
            'pooling_strategy': self.pooling_strategy,
            'normalize_embeddings': self.normalize_embeddings
        }

        # Save to pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"Index saved to {filepath}")

        # Optionally save document IDs separately for reference
        id_list = [doc['id'] for doc in self.documents]
        id_path = os.path.join(self.index_dir, f"{self.__class__.__name__}_doc_ids.json")
        with open(id_path, 'w') as f:
            json.dump(id_list, f)

    def load_index(self, filename: str = None) -> bool:
        """
        Load the index from disk, including embeddings and full document data.

        Args:
            filename: Name of the file to load the index from (default: retriever class name)

        Returns:
            True if index was loaded successfully, False otherwise.
        """
        if not self.index_dir:
            raise ValueError("Index directory not specified")

        if filename is None:
            filename = f"{self.__class__.__name__}.pkl"

        filepath = os.path.join(self.index_dir, filename)

        if not os.path.exists(filepath):
            logger.warning(f"Index file not found: {filepath}")
            return False

        try:
            # Load the saved data
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)

            # Restore embeddings
            self.embeddings = save_data['embeddings']

            # Restore documents: try "documents" first, then fall back to "documents_metadata"
            if 'documents' in save_data:
                self.documents = save_data['documents']
            elif 'documents_metadata' in save_data:
                self.documents = save_data['documents_metadata']
            else:
                logger.warning("No documents metadata found in saved index; setting documents to empty list.")
                self.documents = []

            # Restore other settings if available
            if 'model_path' in save_data:
                self.model_path = save_data['model_path']
            if 'use_faiss' in save_data:
                self.use_faiss = save_data['use_faiss']
            if 'max_length' in save_data:
                self.max_length = save_data['max_length']
            if 'pooling_strategy' in save_data:
                self.pooling_strategy = save_data['pooling_strategy']
            if 'normalize_embeddings' in save_data:
                self.normalize_embeddings = save_data['normalize_embeddings']

            # Rebuild the FAISS index from embeddings
            if self.use_faiss:
                self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
                embeddings_arr = np.array(self.embeddings, dtype=np.float32, copy=True, order='C')

                if self.normalize_embeddings:
                    # Manual L2 normalization to avoid faiss.normalize_L2 issues
                    norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
                    mask = (norms > 1e-10).reshape(-1)  # Avoid division by zero

                    if mask.all():  # All norms are valid
                        normalized_embeddings = embeddings_arr / norms
                        self.faiss_index.add(normalized_embeddings)
                    else:
                        logger.warning(
                            f"Found {(~mask).sum()} embeddings with near-zero norm. Using unnormalized embeddings.")
                        self.faiss_index.add(embeddings_arr)
                else:
                    self.faiss_index.add(embeddings_arr)

            self.is_indexed = True
            logger.info(f"Successfully loaded index from {filepath}")
            logger.info(f"Loaded {len(self.documents)} documents and {self.embeddings.shape[0]} embeddings")
            return True

        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search the CodeBERT index for the given query.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of retrieved documents with relevance scores
        """
        if not self.is_indexed:
            raise ValueError("No index available. Call index() or load_index() first.")

        # Encode query
        query_embedding = self._encode_text(query)

        # Search using FAISS if enabled
        if self.use_faiss and self.faiss_index is not None:
            # Normalize query embedding for cosine similarity if requested
            if self.normalize_embeddings:
                query_embedding_normalized = query_embedding.copy()
                faiss.normalize_L2(np.array([query_embedding_normalized]))
                # Search index
                scores, indices = self.faiss_index.search(np.array([query_embedding_normalized]), k)
            else:
                # Search index
                scores, indices = self.faiss_index.search(np.array([query_embedding]), k)

            # Prepare results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.documents):  # Check if index is valid
                    doc = self.documents[idx]
                    results.append({
                        'id': doc['id'],
                        'score': float(scores[0][i]),
                        'code': doc.get('code', ''),
                        'func_name': doc.get('func_name', ''),
                        'repo': doc.get('repo', ''),
                        'path': doc.get('path', ''),
                        'docstring': doc.get('docstring', '')
                    })
        else:
            # Calculate cosine similarity manually
            if self.normalize_embeddings:
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
                normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                scores = np.dot(normalized_embeddings, query_embedding)
            else:
                scores = np.dot(self.embeddings, query_embedding)

            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:k]

            # Prepare results
            results = []
            for idx in top_indices:
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
            Dictionary with embeddings and related data
        """
        return {
            'embeddings': self.embeddings,
            'model_path': self.model_path,  # Fixed: using model_path instead of model_name
            'use_faiss': self.use_faiss,
            'max_length': self.max_length,
            'pooling_strategy': self.pooling_strategy,
            'normalize_embeddings': self.normalize_embeddings  # Added normalized_embeddings flag
        }

    def set_index_data(self, index_data: Dict[str, Any]) -> None:
        """
        Set the index data after deserialization.

        Args:
            index_data: Dictionary with embeddings and related data
        """
        self.embeddings = index_data['embeddings']
        self.model_path = index_data['model_path']  # Fixed: using model_path instead of model_name
        self.use_faiss = index_data['use_faiss']
        self.max_length = index_data['max_length']
        self.pooling_strategy = index_data['pooling_strategy']

        # Handle backward compatibility
        if 'normalize_embeddings' in index_data:
            self.normalize_embeddings = index_data['normalize_embeddings']

        # Rebuild FAISS index if needed
        if self.use_faiss and self.embeddings is not None:
            self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])

            # Normalize embeddings for cosine similarity if requested
            if self.normalize_embeddings:
                normalized_embeddings = self.embeddings.copy()
                faiss.normalize_L2(normalized_embeddings)
                self.faiss_index.add(normalized_embeddings)
            else:
                self.faiss_index.add(self.embeddings)