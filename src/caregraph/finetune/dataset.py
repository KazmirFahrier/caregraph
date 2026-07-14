"""Synthetic clinical-note dataset for problem-list extraction.

Task: given a free-text clinical note, extract the patient's active problems as a
multi-label set from a fixed vocabulary. This is a realistic fine-tuning target
(structured extraction from clinical text) and it is deliberately constructed so a
naive keyword prompt *cannot* solve it: most condition mentions use paraphrases
(e.g. "renal impairment" for CKD, "reduced ejection fraction" for heart failure)
that share no canonical keyword. A fine-tuned model learns these associations from
data; a keyword prompt does not. That is the whole point of the phase.

All notes are synthetic. No real PHI.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

LABELS = ["CKD", "HYPERTENSION", "DIABETES", "AFIB", "HEART_FAILURE"]

# For each label: the canonical keyword(s) a prompt would match, then paraphrases
# that carry NO canonical keyword. Generation favors paraphrases.
_PHRASES: dict[str, dict[str, list[str]]] = {
    "CKD": {
        "canonical": ["chronic kidney disease", "ckd"],
        "paraphrase": ["renal impairment", "reduced kidney function", "worsening nephropathy",
                       "declining glomerular filtration"],
    },
    "HYPERTENSION": {
        "canonical": ["hypertension"],
        "paraphrase": ["elevated blood pressure", "persistently high bp readings",
                       "poorly controlled pressures"],
    },
    "DIABETES": {
        "canonical": ["diabetes"],
        "paraphrase": ["elevated a1c", "chronic hyperglycemia", "insulin-dependent glucose issues"],
    },
    "AFIB": {
        "canonical": ["atrial fibrillation"],
        "paraphrase": ["irregularly irregular rhythm", "af with rapid ventricular response",
                       "new-onset irregular pulse"],
    },
    "HEART_FAILURE": {
        "canonical": ["heart failure"],
        "paraphrase": ["reduced ejection fraction", "signs of volume overload",
                       "decompensated cardiac function"],
    },
}

_FILLER = [
    "The patient presents for routine follow-up.",
    "Vitals reviewed and medications reconciled.",
    "No acute distress noted on exam.",
    "Plan discussed with the patient.",
]


@dataclass
class Example:
    note: str
    labels: set[str]


def _generate(rng: random.Random, n: int, phrases: dict[str, dict[str, list[str]]],
              paraphrase_prob: float) -> list[Example]:
    examples: list[Example] = []
    for _ in range(n):
        k = rng.randint(1, 3)
        present = rng.sample(LABELS, k)
        sentences: list[str] = [rng.choice(_FILLER)]
        for label in present:
            pool = "paraphrase" if rng.random() < paraphrase_prob else "canonical"
            options = phrases[label][pool] or phrases[label]["canonical"]
            sentences.append(f"Assessment notes {rng.choice(options)}.")
        rng.shuffle(sentences)
        examples.append(Example(note=" ".join(sentences), labels=set(present)))
    return examples


def make_dataset(n: int = 240, seed: int = 7, paraphrase_prob: float = 0.75) -> list[Example]:
    """Generate `n` synthetic (note, labels) examples deterministically."""
    return _generate(random.Random(seed), n, _PHRASES, paraphrase_prob)


def split(examples: list[Example], test_frac: float = 0.25) -> tuple[list[Example], list[Example]]:
    cut = int(len(examples) * (1 - test_frac))
    return examples[:cut], examples[cut:]


def make_split(n: int = 240, seed: int = 7, paraphrase_prob: float = 0.75,
               test_frac: float = 0.25) -> tuple[list[Example], list[Example]]:
    """Train/test split that holds one paraphrase per label OUT of training.

    The test set can use every paraphrase, including the held-out ones, so a model
    that merely memorized training phrasings scores below 1.0 — the fine-tuned
    number then reflects generalization to unseen clinical language, not leakage
    (see AUDIT.md L3).
    """
    rng = random.Random(seed)
    train_phrases = {
        lab: {"canonical": g["canonical"], "paraphrase": g["paraphrase"][:-1]}
        for lab, g in _PHRASES.items()
    }
    n_test = int(n * test_frac)
    train = _generate(rng, n - n_test, train_phrases, paraphrase_prob)
    test = _generate(rng, n_test, _PHRASES, paraphrase_prob)  # full pool incl. held-out
    return train, test


def to_sft_jsonl(examples: list[Example]) -> str:
    """Serialize to instruction-tuning JSONL — the format the real SFT trainer consumes."""
    lines = []
    for ex in examples:
        lines.append(json.dumps({
            "instruction": "Extract the active problem list from the clinical note. "
                           f"Respond with a comma-separated subset of: {', '.join(LABELS)}.",
            "input": ex.note,
            "output": ", ".join(sorted(ex.labels)),
        }))
    return "\n".join(lines)
