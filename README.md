# Code Retrieval and Explanation System for IS 4200/CS 6200 class

This repository contains a modular code retrieval system that implements multiple retrieval approaches and evaluates their performance on code search tasks. The system uses various techniques including lexical matching (BM25), structural code understanding (AST), and semantic embeddings (CodeBERT), as well as hybrid combinations of these approaches.

## Dataset

This project uses the CodeSearchNet dataset, which contains pairs of code snippets and their natural language descriptions. You need to download the dataset from:

[CodeSearchNet Dataset on Kaggle](https://www.kaggle.com/datasets/omduggineni/codesearchnet)

After downloading, extract the dataset to a directory named `data` in the root of the project.

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
project/
├── data/                    # CodeSearchNet dataset directory
│   └── python/              # Python subset of the dataset
├── models/                  # Directory for saved models
│   └── codebert-finetuned/  # Fine-tuned CodeBERT model
├── preprocessing.py         # Code preprocessing utilities
├── evaluation.py            # Main evaluation script
├── retrieval/               # Retrieval implementations
│   ├── base.py              # Base retriever class
│   ├── bm25.py              # BM25 retriever
│   ├── ast_based.py         # AST-based retriever
│   └── codebert/            # CodeBERT-based retrieval
│       ├── codebert.py      # CodeBERT retriever implementation
│       └── finetuning.py    # Script for fine-tuning CodeBERT
└── results/                 # Evaluation results
```

## Usage

### Running Evaluation

The main script for running evaluations is `evaluation.py`. Here's how to use it:

```bash
python evaluation.py --data_dir ./data/python --subset valid --retrievers bm25 ast codebert bm25+ast codebert+bm25 codebert+ast codebert+bm25+ast
```

#### Command Line Arguments

- `--data_dir`: Directory containing CodeSearchNet data (default: `./data/python`)
- `--subset`: Data subset to use (`train`, `valid`, or `test`) (default: `valid`)
- `--output_dir`: Directory to save results (default: `./results`)
- `--tokenization`: Tokenization method for BM25 (`simple`, `subtokens`, `ast_based`) (default: `subtokens`)
- `--bm25_variant`: BM25 variant to use (`okapi`, `plus`) (default: `okapi`)
- `--retrievers`: Retrievers to evaluate (default: `bm25`)
  - Options: `bm25`, `ast`, `codebert`, `bm25+ast`, `codebert+bm25`, `codebert+ast`, `codebert+bm25+ast`
- `--num_queries`: Number of queries to evaluate (default: `100`, `0` for all)
- `--k`: Number of results to retrieve per query (default: `10`)

### Fine-tuning CodeBERT

Before using CodeBERT, you might want to fine-tune it on the CodeSearchNet data:

```bash
python retrieval/codebert/finetuning.py --data_dir ./data/python --output_dir ./models/codebert-finetuned --num_files 6
```

The fine-tuning uses contrastive learning to optimize CodeBERT for code search.

NOTE: If you are planning to use GPU and build FAISS index after fine-tuning, please install the `faiss` relevant to your GPU hardware (not `faiss-cpu`).

### Understanding the Results

After running an evaluation, results are saved to the specified output directory (default: `./results`). The results include:

- Detailed metrics in JSON format
- Markdown reports for each retriever
- Comparison report for all evaluated retrievers
- Visualizations:
  - Position distribution plots
  - Query time distribution plots
  - Comparison bar charts for key metrics

## Retrieval Methods

The system implements several retrieval approaches:

### BM25 (Lexical Matching)

Uses the BM25 algorithm (Okapi variant) to perform lexical matching between queries and code. Supports different tokenization methods.

### AST-based (Structural Understanding)

Extracts features from code's Abstract Syntax Tree to capture structural patterns regardless of variable names or coding style.

### CodeBERT (Semantic Understanding)

Uses the CodeBERT transformer model to generate semantic embeddings for code and queries, which are then compared using cosine similarity.

### Hybrid Approaches

Combines multiple retrieval methods using a weighted sum of normalized scores:

- `bm25+ast`: Combines lexical matching with structural understanding
- `codebert+bm25`: Combines semantic and lexical matching
- `codebert+ast`: Combines semantic and structural understanding
- `codebert+bm25+ast`: Combines all three approaches

## Evaluation Metrics

The system evaluates retrieval performance using standard information retrieval metrics:

- Precision@k (k=1,3,5,10): Proportion of relevant results in the top-k
- Recall@k: Proportion of relevant results found in the top-k
- MRR (Mean Reciprocal Rank): Average position of the first relevant result
- MAP (Mean Average Precision): Average precision across all relevant results
- Query execution time

