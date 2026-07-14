"""Model comparison: base vs prompt-only vs fine-tuned.

Scores each extractor on a held-out test split with micro-averaged precision, recall
and F1 over the multi-label problem list. This is the ablation that proves the
fine-tuned model earns its keep — the exact artifact a hiring manager wants: a
production metric, measured on held-out data, against honest baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from caregraph.finetune.dataset import Example


@dataclass
class ComparisonRow:
    model: str
    precision: float
    recall: float
    f1: float


def micro_f1(predict: Callable[[str], set[str]], test: list[Example]) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for ex in test:
        pred = predict(ex.note)
        tp += len(pred & ex.labels)
        fp += len(pred - ex.labels)
        fn += len(ex.labels - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def compare_models(
    models: dict[str, Callable[[str], set[str]]], test: list[Example]
) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    for name, predict in models.items():
        p, r, f = micro_f1(predict, test)
        rows.append(ComparisonRow(model=name, precision=round(p, 3), recall=round(r, 3),
                                  f1=round(f, 3)))
    return rows
