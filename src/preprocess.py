"""
Data-loading helper.  For the smoke test we just stream the first `n_samples`
of GSM8K training set via huggingface-datasets.  The full experiment streams
all samples lazily so that we stay well below the 500 MB RAM limit.
"""
from __future__ import annotations

import logging
from typing import Dict, Any

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_dataset_from_config(cfg: Dict[str, Any], split: str):
    name = cfg.get("dataset", "openai/gsm8k")
    streaming = cfg.get("streaming", True)
    logger.info("Loading %s split=%s streaming=%s", name, split, streaming)
    ds = load_dataset(name, "main", split=split, streaming=streaming)
    if isinstance(ds, list):  # non-streaming mode returns Dataset
        return ds
    return ds.take(cfg.get("n_samples", 256)) if cfg.get("n_samples") else ds
