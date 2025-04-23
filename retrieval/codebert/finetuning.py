"""
finetuning.py

Script for fine-tuning CodeBERT model for code search tasks in Google Colab.
Uses contrastive learning to optimize embeddings for code-query matching.
"""

import os
import json
import time
import random
import numpy as np
import torch
torch.cuda.empty_cache()
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    get_linear_schedule_with_warmup,
    set_seed
)
from torch.optim import AdamW
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import sys


import gc
gc.collect()

# Use print statements with flush=True so messages appear immediately
def info(msg):
    print(f"[INFO] {msg}", flush=True)

def warning(msg):
    print(f"[WARNING] {msg}", flush=True)

def error(msg):
    print(f"[ERROR] {msg}", flush=True)

# Dataset for code search task
class CodeSearchDataset(Dataset):
    def __init__(self, tokenizer, code_samples, nl_samples, max_length=512, code_prefix="", nl_prefix=""):
        self.tokenizer = tokenizer
        self.code_samples = code_samples
        self.nl_samples = nl_samples
        self.max_length = max_length
        self.code_prefix = code_prefix
        self.nl_prefix = nl_prefix
        assert len(code_samples) == len(nl_samples), "Code and NL samples must have the same length"

    def __len__(self):
        return len(self.code_samples)

    def __getitem__(self, idx):
        code = self.code_prefix + self.code_samples[idx]
        nl = self.nl_prefix + self.nl_samples[idx]
        code_tokens = self.tokenizer(
            code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        nl_tokens = self.tokenizer(
            nl,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "code_input_ids": code_tokens["input_ids"].squeeze(),
            "code_attention_mask": code_tokens["attention_mask"].squeeze(),
            "nl_input_ids": nl_tokens["input_ids"].squeeze(),
            "nl_attention_mask": nl_tokens["attention_mask"].squeeze(),
        }

# CodeBERT model for contrastive learning
class CodeBERTForContrastiveLearning(torch.nn.Module):
    def __init__(self, model_name_or_path, pooling_strategy="mean"):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name_or_path)
        self.pooling_strategy = pooling_strategy
        self.temperature = 0.05  # Temperature parameter for contrastive loss

    def forward(self, code_input_ids, code_attention_mask, nl_input_ids, nl_attention_mask, return_embeddings=False):
        code_outputs = self.roberta(
            input_ids=code_input_ids,
            attention_mask=code_attention_mask
        )
        code_hidden_states = code_outputs.last_hidden_state

        nl_outputs = self.roberta(
            input_ids=nl_input_ids,
            attention_mask=nl_attention_mask
        )
        nl_hidden_states = nl_outputs.last_hidden_state

        if self.pooling_strategy == "cls":
            code_embeddings = code_hidden_states[:, 0, :]
            nl_embeddings = nl_hidden_states[:, 0, :]
        elif self.pooling_strategy == "max":
            code_attention_mask_expanded = code_attention_mask.unsqueeze(-1).expand(code_hidden_states.size()).float()
            code_hidden_states = code_hidden_states * code_attention_mask_expanded
            code_embeddings = torch.max(code_hidden_states, dim=1)[0]
            nl_attention_mask_expanded = nl_attention_mask.unsqueeze(-1).expand(nl_hidden_states.size()).float()
            nl_hidden_states = nl_hidden_states * nl_attention_mask_expanded
            nl_embeddings = torch.max(nl_hidden_states, dim=1)[0]
        else:  # mean pooling
            code_attention_mask_expanded = code_attention_mask.unsqueeze(-1).expand(code_hidden_states.size()).float()
            code_sum_embeddings = torch.sum(code_hidden_states * code_attention_mask_expanded, dim=1)
            code_sum_mask = torch.sum(code_attention_mask_expanded, dim=1)
            code_embeddings = code_sum_embeddings / code_sum_mask.clamp(min=1e-9)
            nl_attention_mask_expanded = nl_attention_mask.unsqueeze(-1).expand(nl_hidden_states.size()).float()
            nl_sum_embeddings = torch.sum(nl_hidden_states * nl_attention_mask_expanded, dim=1)
            nl_sum_mask = torch.sum(nl_attention_mask_expanded, dim=1)
            nl_embeddings = nl_sum_embeddings / nl_sum_mask.clamp(min=1e-9)

        code_embeddings = torch.nn.functional.normalize(code_embeddings, p=2, dim=1)
        nl_embeddings = torch.nn.functional.normalize(nl_embeddings, p=2, dim=1)

        if return_embeddings:
            return code_embeddings, nl_embeddings

        scores = torch.matmul(nl_embeddings, code_embeddings.t()) / self.temperature
        labels = torch.arange(scores.size(0), device=scores.device)
        loss = torch.nn.CrossEntropyLoss()(scores, labels)
        return loss, scores

def load_jsonl_data(file_paths: List[str]) -> List[Dict[str, Any]]:
    data = []
    for file_path in file_paths:
        info(f"Loading data from {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        data.append(sample)
                    except json.JSONDecodeError:
                        warning(f"Failed to parse line in {file_path}")
        except Exception as e:
            error(f"Failed to load {file_path}: {e}")
    info(f"Loaded {len(data)} samples")
    return data

def prepare_data(data: List[Dict[str, Any]], code_key: str = "code", nl_key: str = "docstring", min_nl_length: int = 10, max_samples: int = None) -> Tuple[List[str], List[str]]:
    code_samples = []
    nl_samples = []
    for sample in data:
        code = sample.get(code_key, "")
        nl = sample.get(nl_key, "")
        if len(code.strip()) < 10 or len(nl.strip()) < min_nl_length:
            continue
        code_samples.append(code)
        nl_samples.append(nl)
        if max_samples and len(code_samples) >= max_samples:
            break
    return code_samples, nl_samples

def train_codebert(
    train_data_dir = os.path.join("python", "python", "final", "jsonl", "train"),
    val_data_dir = os.path.join("python", "python", "final", "jsonl", "valid"),
    output_dir = os.path.join("models", "codebert-finetuned"),
    num_train_files = 10,
    num_val_files = 1,
    min_nl_length = 10,
    max_samples = 20000,      # Maximum training samples
    max_val_samples = None,   # Maximum validation samples (can set to a fixed number as used in evaluation)
    model_name_or_path = "microsoft/codebert-base",
    max_length = 512,
    pooling_strategy = "mean",
    code_prefix = "",
    nl_prefix = "",
    batch_size = 16,
    num_train_epochs = 3,
    learning_rate = 2e-5,
    weight_decay = 0.01,
    adam_epsilon = 1e-8,
    warmup_ratio = 0.1,
    save_steps = 1000,
    seed = 42,
    num_workers = 4,
    no_cuda = False
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    info(f"Loading tokenizer from {model_name_or_path}")
    tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)

    # Load full training set (no splitting)
    train_data_files = []
    for i in range(num_train_files):
        file_path = os.path.join(train_data_dir, f"python_train_{i}.jsonl")
        if os.path.exists(file_path):
            train_data_files.append(file_path)
        else:
            warning(f"Training file not found: {file_path}")

    if not train_data_files:
        raise ValueError(f"No training data files found in {train_data_dir} matching the pattern python_train_*.jsonl")

    train_data = load_jsonl_data(train_data_files)
    train_code, train_nl = prepare_data(train_data, min_nl_length=min_nl_length, max_samples=max_samples)
    info(f"Prepared {len(train_code)} training samples")

    # Load separate validation set (e.g. evaluation set)
    val_data_files = []
    for i in range(num_val_files):
        file_path = os.path.join(val_data_dir, f"python_valid_{i}.jsonl")
        if os.path.exists(file_path):
            val_data_files.append(file_path)
        else:
            warning(f"Validation file not found: {file_path}")

    if not val_data_files:
        raise ValueError(f"No validation data files found in {val_data_dir} matching the pattern python_valid_*.jsonl")

    val_data = load_jsonl_data(val_data_files)
    val_code, val_nl = prepare_data(val_data, min_nl_length=min_nl_length, max_samples=max_val_samples)
    info(f"Prepared {len(val_code)} validation samples")

    # Create datasets
    train_dataset = CodeSearchDataset(tokenizer=tokenizer, code_samples=train_code, nl_samples=train_nl, max_length=max_length, code_prefix=code_prefix, nl_prefix=nl_prefix)
    val_dataset = CodeSearchDataset(tokenizer=tokenizer, code_samples=val_code, nl_samples=val_nl, max_length=max_length, code_prefix=code_prefix, nl_prefix=nl_prefix)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    info(f"Initializing model from {model_name_or_path}")
    model = CodeBERTForContrastiveLearning(model_name_or_path=model_name_or_path, pooling_strategy=pooling_strategy)

    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    total_steps = len(train_dataloader) * num_train_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    info("***** Starting training *****")
    info(f"  Num training examples = {len(train_dataset)}")
    info(f"  Num Epochs = {num_train_epochs}")
    info(f"  Batch size = {batch_size}")
    info(f"  Total optimization steps = {total_steps}")

    global_step = 0
    best_val_loss = float("inf")
    best_model_path = None
    train_losses = []
    val_losses = []

    for epoch in range(num_train_epochs):
        info(f"Epoch {epoch + 1}/{num_train_epochs}")
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Training (Epoch {epoch + 1})")
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss, _ = model(
                code_input_ids=batch["code_input_ids"],
                code_attention_mask=batch["code_attention_mask"],
                nl_input_ids=batch["nl_input_ids"],
                nl_attention_mask=batch["nl_attention_mask"]
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1
            if save_steps > 0 and global_step % save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.roberta.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                info(f"Saved checkpoint to {checkpoint_dir}")

        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        info(f"  Average train loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        val_mrr = 0
        progress_bar = tqdm(val_dataloader, desc=f"Validation (Epoch {epoch + 1})")
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                loss, scores = model(
                    code_input_ids=batch["code_input_ids"],
                    code_attention_mask=batch["code_attention_mask"],
                    nl_input_ids=batch["nl_input_ids"],
                    nl_attention_mask=batch["nl_attention_mask"]
                )
                val_loss += loss.item()
                rankings = torch.argsort(scores, dim=1, descending=True)
                for i in range(rankings.size(0)):
                    rank = (rankings[i] == i).nonzero(as_tuple=True)[0].item() + 1
                    val_mrr += 1.0 / rank
            progress_bar.set_postfix({"loss": loss.item()})
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_mrr = val_mrr / len(val_dataset)
        val_losses.append(avg_val_loss)
        info(f"  Average validation loss: {avg_val_loss:.4f}")
        info(f"  Validation MRR: {avg_val_mrr:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(output_dir, "best_model")
            os.makedirs(best_model_path, exist_ok=True)
            model.roberta.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            with open(os.path.join(best_model_path, "training_info.json"), "w") as f:
                json.dump({
                    "best_val_loss": best_val_loss,
                    "best_val_mrr": avg_val_mrr,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "pooling_strategy": pooling_strategy
                }, f, indent=2)
            info(f"Saved best model to {best_model_path}")

    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    model.roberta.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs": num_train_epochs,
            "steps": global_step
        }, f, indent=2)

    info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    info(f"Best model saved to: {best_model_path}")
    info(f"Final model saved to: {final_model_path}")

    return best_model_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Fine-tune CodeBERT for code-search with contrastive learning."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing `train/` and `valid/` JSONL shards."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save checkpoints and the final model."
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=10,
        help="Number of *training* JSONL shards to load (python_train_0.jsonl …)."
    )
    parser.add_argument(
        "--num_val_files",
        type=int,
        default=1,
        help="Number of *validation* JSONL shards to load (python_valid_0.jsonl …)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Mini-batch size."
    )
    args = parser.parse_args()

    start = time.time()
    best_model_path = train_codebert(
        train_data_dir=os.path.join(args.data_dir, "train"),
        val_data_dir=os.path.join(args.data_dir, "valid"),
        output_dir=args.output_dir,
        num_train_files=args.num_files,
        num_val_files=args.num_val_files,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - start
    info(f"Finished in {elapsed/3600:.2f} h. Best model at {best_model_path}")

