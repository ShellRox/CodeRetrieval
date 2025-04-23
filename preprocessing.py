"""
preprocessing.py

This module handles all preprocessing operations for code retrieval,
ensuring consistent normalization across different retrieval methods.
"""

import re
import ast
import json
import os
from typing import List, Dict, Union, Any, Optional, Tuple
import tokenize
from io import StringIO


def normalize_code(code: str) -> str:
    """
    Normalize code by removing comments, extra whitespace, and standardizing line endings.

    Args:
        code: Raw code string to normalize

    Returns:
        Normalized code string
    """
    # Remove comments and standardize whitespace
    try:
        # Try to tokenize the Python code to properly handle comments
        tokens = []
        token_generator = tokenize.generate_tokens(StringIO(code).readline)
        for token in token_generator:
            if token.type != tokenize.COMMENT:
                tokens.append(token)

        # Reconstruct code without comments
        result = tokenize.untokenize(tokens)
        # Normalize whitespace
        result = re.sub(r'\s+', ' ', result)
        # Normalize line endings
        result = result.replace('\r\n', '\n').replace('\r', '\n')
        return result.strip()
    except Exception:
        # Fallback if tokenization fails
        # Remove single line comments
        code = re.sub(r'#.*', '', code)
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        return code.strip()


def tokenize_code(code: str, method: str = 'simple') -> List[str]:
    """
    Tokenize code using specified method.

    Args:
        code: Code string to tokenize
        method: Tokenization method ('simple', 'subtokens', 'ast_based')

    Returns:
        List of tokens
    """
    if method == 'simple':
        # Simple whitespace and punctuation tokenization
        # First normalize the code
        code = normalize_code(code)
        # Replace punctuation with spaces
        code = re.sub(r'([^\w\s])', r' \1 ', code)
        # Split by whitespace and filter out empty tokens
        tokens = [token for token in code.split() if token]
        return tokens

    elif method == 'subtokens':
        # Split identifiers like camelCase, snake_case, etc.
        # First get simple tokens
        simple_tokens = tokenize_code(code, 'simple')

        result = []
        for token in simple_tokens:
            # Split by camelCase
            camel_case_tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', token)
            # Split by snake_case
            snake_case_tokens = []
            for camel_token in camel_case_tokens:
                snake_case_tokens.extend(camel_token.split('_'))

            # Add original token and all subtokens
            result.append(token)
            result.extend([t.lower() for t in snake_case_tokens if t and len(t) > 1])

        return result

    elif method == 'ast_based':
        # This is a placeholder - we'll implement proper AST-based tokenization later
        # For now, we'll just use simple tokenization
        return tokenize_code(code, 'simple')

    else:
        raise ValueError(f"Unknown tokenization method: {method}")


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries, each representing a JSON object
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_codesearchnet_data(base_dir: str, subset: str = 'train', limit: int = None) -> List[Dict[str, Any]]:
    """
    Load CodeSearchNet data from the specified directory.

    Args:
        base_dir: Base directory containing the data
        subset: Data subset ('train', 'valid', or 'test')
        limit: Maximum number of examples to load (None for all)

    Returns:
        List of code examples with their metadata
    """
    data = []

    # Find all relevant JSONL files
    pattern = f"python_{subset}_*.jsonl"
    jsonl_files = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith(f"python_{subset}_") and file.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, file))

    # Sort files to ensure deterministic loading
    jsonl_files.sort()

    # Load data from each file
    count = 0
    for file_path in jsonl_files:
        file_data = load_jsonl(file_path)
        for item in file_data:
            data.append(item)
            count += 1
            if limit is not None and count >= limit:
                return data

    return data


def extract_functions_from_code(code: str) -> List[Dict[str, str]]:
    """
    Extract individual functions from a Python file.

    Args:
        code: Python code string

    Returns:
        List of dictionaries with function name and code
    """
    functions = []

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get function source code
                lineno = node.lineno
                end_lineno = node.end_lineno if hasattr(node, 'end_lineno') else None

                # If end_lineno is not available (Python < 3.8),
                # we'll just extract the whole function declaration
                func_code = ast.get_source_segment(code, node)

                if func_code:
                    functions.append({
                        'name': node.name,
                        'code': func_code,
                        'lineno': lineno,
                        'end_lineno': end_lineno
                    })
    except SyntaxError:
        # If the code has syntax errors, we cannot reliably extract functions
        # Just return an empty list
        pass

    return functions


def create_code_document(code_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a standardized document from code data for indexing.

    Args:
        code_data: Raw code data (e.g., from CodeSearchNet)

    Returns:
        Standardized document with preprocessed fields
    """
    # Extract necessary fields
    code = code_data.get('code', '')
    func_name = code_data.get('func_name', '')
    docstring = code_data.get('docstring', '')
    repo = code_data.get('repo', '')
    path = code_data.get('path', '')

    # Extract individual functions if this is a full file
    functions = extract_functions_from_code(code) if len(code.strip().split('\n')) > 5 else []

    # Create standardized document
    document = {
        'id': f"{repo}/{path}/{func_name}" if repo and path and func_name else str(hash(code)),
        'code': code,
        'normalized_code': normalize_code(code),
        'func_name': func_name,
        'docstring': docstring,
        'repo': repo,
        'path': path,
        'functions': functions,
        'tokens': {
            'simple': tokenize_code(code, 'simple'),
            'subtokens': tokenize_code(code, 'subtokens')
        }
    }

    return document