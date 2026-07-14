"""Problem-list extractors: two baselines and one genuinely trainable model.

- `MajorityBaseline`  : the untrained "base model" reference — predicts only labels
                        with >50% base rate (usually none). Establishes the floor.
- `KeywordExtractor`  : the prompt-only baseline — matches canonical keywords. Good
                        precision, poor recall because most notes use paraphrases.
- `LogisticExtractor` : a per-label logistic regression over bag-of-words trained with
                        SGD. Pure Python, deterministic, no dependencies — it actually
                        *learns* the paraphrase->label associations from the data.

`LogisticExtractor` is the offline stand-in that makes the fine-tuning story testable
in CI. `finetune/lora.py` is the production GPU path (real PEFT/QLoRA on an LLM); both
implement the same `predict(note) -> set[str]` contract, so `compare.py` scores them
identically.
"""
from __future__ import annotations

import math
import re
from typing import Protocol

from caregraph.finetune.dataset import LABELS, Example, _PHRASES

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class Extractor(Protocol):
    name: str

    def predict(self, note: str) -> set[str]: ...


class MajorityBaseline:
    name = "base (untrained)"

    def __init__(self) -> None:
        self._always: set[str] = set()

    def fit(self, examples: list[Example]) -> "MajorityBaseline":
        n = len(examples) or 1
        counts = {lab: 0 for lab in LABELS}
        for ex in examples:
            for lab in ex.labels:
                counts[lab] += 1
        self._always = {lab for lab, c in counts.items() if c / n > 0.5}
        return self

    def predict(self, note: str) -> set[str]:
        return set(self._always)


class KeywordExtractor:
    name = "prompt-only (keywords)"

    def predict(self, note: str) -> set[str]:
        text = note.lower()
        found: set[str] = set()
        for label, groups in _PHRASES.items():
            if any(kw in text for kw in groups["canonical"]):
                found.add(label)
        return found


class LogisticExtractor:
    name = "fine-tuned (LoRA proxy)"

    def __init__(self, lr: float = 0.5, epochs: int = 60, l2: float = 1e-4) -> None:
        self._lr = lr
        self._epochs = epochs
        self._l2 = l2
        self._vocab: dict[str, int] = {}
        self._w: dict[str, list[float]] = {}
        self._b: dict[str, float] = {}

    def fit(self, examples: list[Example]) -> "LogisticExtractor":
        # Build vocabulary from the training notes only.
        for ex in examples:
            for tok in set(_tokens(ex.note)):
                self._vocab.setdefault(tok, len(self._vocab))
        dim = len(self._vocab)
        self._w = {lab: [0.0] * dim for lab in LABELS}
        self._b = {lab: 0.0 for lab in LABELS}

        feats = [self._features(ex.note) for ex in examples]
        for _ in range(self._epochs):
            for ex, x in zip(examples, feats):
                for lab in LABELS:
                    y = 1.0 if lab in ex.labels else 0.0
                    z = self._b[lab] + sum(self._w[lab][i] for i in x)
                    p = _sigmoid(z)
                    g = p - y
                    self._b[lab] -= self._lr * g
                    for i in x:
                        self._w[lab][i] -= self._lr * (g + self._l2 * self._w[lab][i])
        return self

    def predict(self, note: str) -> set[str]:
        x = self._features(note)
        out: set[str] = set()
        for lab in LABELS:
            z = self._b[lab] + sum(self._w[lab][i] for i in x)
            if _sigmoid(z) >= 0.5:
                out.add(lab)
        return out

    def _features(self, note: str) -> list[int]:
        idx = {self._vocab[t] for t in _tokens(note) if t in self._vocab}
        return sorted(idx)


def _sigmoid(z: float) -> float:
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))
