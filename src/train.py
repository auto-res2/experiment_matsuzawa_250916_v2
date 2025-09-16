"""
train.py
---------
Contains the "training" logic.  In this minimal refactor we only fine-tune a
MiniLM encoder for one epoch on the optimisation examples that are provided by
`src.preprocess.load_dataset(...)`.  The routine is intentionally light-weight
so that it fits into the 500 MB RAM / single-T4 constraint of the execution
environment.

If you want to plug-in the full HiMAP surrogate later on you only have to
replace the tiny fine-tune loop below – the public interface (`train` and
`load_model`) should remain stable.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AdamW,
)

logger = logging.getLogger(__name__)

def _freeze_except_head(model: torch.nn.Module, n_unfrozen: int = 1):
    """Freeze all encoder layers except the last `n_unfrozen`."""
    if not hasattr(model, "encoder"):
        return  # e.g. DistilBERT; nothing to do
    layers = list(model.encoder.layer)
    for layer in layers[:-n_unfrozen]:
        for p in layer.parameters():
            p.requires_grad = False


def train(config: Dict[str, Any], dataset):
    """Fine-tunes the encoder for *one* epoch on the optimisation set.

    Args
    -----
    config : Dict[str, Any]
        The loaded YAML configuration sub-tree under the key `model`.
    dataset : datasets.Dataset
        The optimisation dataset coming from `src.preprocess`.
    Returns
    -------
    model : torch.nn.Module – fine-tuned encoder ready for evaluation.
    metrics : Dict[str, float] – training loss so that the caller can log it.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = config["name"]
    lr = config.get("lr", 1e-5)
    num_epochs = int(config.get("epochs", 1))
    batch_size = int(config.get("batch_size", 8))
    unfreeze_last_n = int(config.get("unfreeze_last_n", 1))

    logger.info("Loading %s …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, device_map="auto")
    _freeze_except_head(model, n_unfrozen=unfreeze_last_n)
    model.to(device)

    # ───────────────────────────────────────────────────────────────── training
    def collate(batch):
        texts = [x["question"] for x in batch]
        return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    optimiser = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimiser, num_warmup_steps=5, num_training_steps=len(loader) * num_epochs
    )

    model.train()
    total_loss = 0.0
    for _ in range(num_epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, return_dict=True)
            # Self-supervised: minimise L2 norm of CLS embeddings (dummy objective)
            loss = (outputs.last_hidden_state[:, 0, :] ** 2).mean()
            loss.backward()
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad(set_to_none=True)
            total_loss += loss.item()
    avg_loss = total_loss / (len(loader) * num_epochs)
    return model, {"train_loss": avg_loss}


def save_model(model: torch.nn.Module, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)


def load_model(model_dir: str):
    return AutoModel.from_pretrained(model_dir, device_map="auto")
