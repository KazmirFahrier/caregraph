"""Phase-3 demo: fine-tuning ablation. Runs offline in seconds, no GPU.

Trains the LoRA-proxy extractor on synthetic clinical notes and compares it against
the untrained base and the prompt-only keyword baseline on a held-out split.

    python scripts/finetune_demo.py
"""
from __future__ import annotations

from caregraph.finetune import (
    KeywordExtractor,
    LogisticExtractor,
    MajorityBaseline,
    compare_models,
)
from caregraph.finetune.dataset import make_split, to_sft_jsonl


def main() -> None:
    # Held-out split: one paraphrase per label is kept out of training, so the
    # fine-tuned score reflects generalization to unseen phrasings, not memorization.
    train, test = make_split(n=240, seed=7, test_frac=0.25)

    base = MajorityBaseline().fit(train)
    keyword = KeywordExtractor()
    tuned = LogisticExtractor().fit(train)

    print("=" * 72)
    print("CareGraph — Phase 3 fine-tuning ablation (problem-list extraction)")
    print("=" * 72)
    print(f"train={len(train)}  test={len(test)}  labels are multi-label per note")
    print("\nExample SFT record (what the QLoRA trainer consumes):")
    print("  " + to_sft_jsonl(train[:1]))

    rows = compare_models({
        base.name: base.predict,
        keyword.name: keyword.predict,
        tuned.name: tuned.predict,
    }, test)

    print(f"\n{'model':<28}{'precision':>10}{'recall':>10}{'f1':>8}")
    print("-" * 56)
    for r in rows:
        print(f"{r.model:<28}{r.precision:>10.3f}{r.recall:>10.3f}{r.f1:>8.3f}")

    best = max(rows, key=lambda r: r.f1)
    print(f"\nBest: {best.model} (F1={best.f1:.3f}). The keyword prompt misses paraphrased")
    print("mentions ('renal impairment', 'reduced ejection fraction'); the fine-tuned")
    print("model learns them from data. Real GPU path: caregraph/finetune/lora.py")


if __name__ == "__main__":
    main()
