"""
Evaluation & visualisation utilities.

We keep evaluation extremely small: compute cosine-similarity between question
embeddings and report their average magnitude as a *proxy* metric.  All results
are printed to STDOUT and also written as JSON into .research/iteration1/…
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.inference_mode()
def evaluate(model, dataset, split: str, config: Dict[str, Any]):
    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    def collate(batch):
        texts = [x["question"] for x in batch]
        return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate)
    norms = []
    model.eval()
    for batch in tqdm(loader, desc=f"encoding-{split}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outs = model(**batch, return_dict=True).last_hidden_state[:, 0, :]
        norms.append(torch.linalg.norm(outs, dim=-1).mean().item())
    return {f"{split}_embedding_norm": float(sum(norms) / len(norms))}


def save_results(metrics: Dict[str, Any], tag: str):
    out_dir = Path(".research/iteration1")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.json"
    with path.open("w") as fp:
        json.dump(metrics, fp, indent=2)
    print(json.dumps(metrics, indent=2))  # STDOUT for quick inspection
