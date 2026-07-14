from caregraph.finetune import (
    KeywordExtractor,
    LogisticExtractor,
    MajorityBaseline,
    micro_f1,
)
from caregraph.finetune.dataset import LABELS, make_dataset, make_split, to_sft_jsonl


def _data():
    return make_split(n=240, seed=7, test_frac=0.25)


def test_dataset_is_deterministic_and_multilabel() -> None:
    a = make_dataset(n=50, seed=7)
    b = make_dataset(n=50, seed=7)
    assert [e.note for e in a] == [e.note for e in b]
    assert all(e.labels and e.labels <= set(LABELS) for e in a)


def test_sft_jsonl_has_expected_fields() -> None:
    import json
    train, _ = _data()
    rec = json.loads(to_sft_jsonl(train[:1]))
    assert set(rec) == {"instruction", "input", "output"}


def test_finetuned_beats_baselines() -> None:
    train, test = _data()
    base = MajorityBaseline().fit(train)
    keyword = KeywordExtractor()
    tuned = LogisticExtractor().fit(train)

    _, _, f_base = micro_f1(base.predict, test)
    _, _, f_kw = micro_f1(keyword.predict, test)
    _, _, f_tuned = micro_f1(tuned.predict, test)

    # The trained model must beat both the untrained base and the prompt-only baseline.
    assert f_tuned > f_kw > f_base
    assert f_tuned >= 0.8


def test_keyword_baseline_has_high_precision_low_recall() -> None:
    train, test = _data()
    p, r, _ = micro_f1(KeywordExtractor().predict, test)
    # Keywords are precise when they fire but miss paraphrased mentions.
    assert p >= 0.9
    assert r < 0.6
