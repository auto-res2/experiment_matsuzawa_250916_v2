"""Command-line entry-point orchestrating smoke-test **and** full experiment.

Example usages (required by the assignment):

uv run python -m src.main --smoke-test
uv run python -m src.main --full-experiment
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from . import preprocess as pp
from . import train
from . import evaluate as ev

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

CFG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_yaml(name: str):
    with open(CFG_DIR / name, "r") as fp:
        return yaml.safe_load(fp)


def run(cfg, tag: str):
    # 1) preprocessing
    train_ds = pp.load_dataset_from_config(cfg, split="train")
    test_ds = pp.load_dataset_from_config(cfg, split="test")

    # 2) training
    model, train_metrics = train.train(cfg["model"], train_ds)

    # 3) evaluation
    test_metrics = ev.evaluate(model, test_ds, "test", cfg)

    # 4) aggregation & persistence
    metrics = {**train_metrics, **test_metrics}
    ev.save_results(metrics, tag)


def main(argv=None):  # noqa: D401 – imperative mood
    parser = argparse.ArgumentParser("HiMAP surrogate – runnable refactor")
    parser.add_argument("--smoke-test", action="store_true", help="run quick check")
    parser.add_argument("--full-experiment", action="store_true", help="run full size")
    args = parser.parse_args(argv)

    if not (args.smoke_test ^ args.full_experiment):
        parser.error("Specify exactly one of --smoke-test / --full-experiment")

    if args.smoke_test:
        cfg = _load_yaml("smoke_test.yaml")
        tag = "smoke_test"
    else:
        cfg = _load_yaml("full_experiment.yaml")
        tag = "full_run"

    try:
        run(cfg, tag)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user – partial results might be missing.")
        sys.exit(130)


if __name__ == "__main__":
    main()
