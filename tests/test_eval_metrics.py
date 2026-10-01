from __future__ import annotations

import pytest
from metrics import compute_metrics

INVALID = "(invalid)"
LABELS = ["a", "b", "c"]


def test_hand_computed_three_class_example() -> None:
    # true:  a a a b b c
    # pred:  a a b b c c
    m = compute_metrics(["a", "a", "a", "b", "b", "c"], ["a", "a", "b", "b", "c", "c"], LABELS, INVALID)
    assert m["n"] == 6 and m["correct"] == 4
    assert m["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    by = {r["label"]: r for r in m["per_label"]}
    # a: TP 2, FP 0, FN 1
    assert (by["a"]["precision"], by["a"]["recall"], by["a"]["f1"], by["a"]["support"]) == (1.0, 0.6667, 0.8, 3)
    # b: TP 1, FP 1, FN 1
    assert (by["b"]["precision"], by["b"]["recall"], by["b"]["f1"], by["b"]["support"]) == (0.5, 0.5, 0.5, 2)
    # c: TP 1, FP 1, FN 0
    assert (by["c"]["precision"], by["c"]["recall"], by["c"]["f1"], by["c"]["support"]) == (0.5, 1.0, 0.6667, 1)
    assert m["macro"] == {"precision": 0.6667, "recall": 0.7222, "f1": 0.6556}
    # weighted by support 3/2/1
    assert m["weighted"]["f1"] == pytest.approx((0.8 * 3 + 0.5 * 2 + (2 / 3) * 1) / 6, abs=1e-4)
    assert m["confusion_labels"] == ["a", "b", "c", INVALID]
    assert m["confusion_matrix"] == [[2, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0]]
    assert m["invalid_predictions"] == 0


def test_invalid_predictions_count_as_wrong_and_unseen_labels_score_zero() -> None:
    m = compute_metrics(["a", "b"], ["a", INVALID], LABELS, INVALID)
    assert m["accuracy"] == 0.5 and m["invalid_predictions"] == 1
    by = {r["label"]: r for r in m["per_label"]}
    assert by["b"]["recall"] == 0.0 and by["b"]["support"] == 1
    assert by["c"] == {"label": "c", "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    assert m["confusion_matrix"][1] == [0, 0, 0, 1]  # true b -> (invalid) column


def test_rejects_unknown_predictions_and_bad_input() -> None:
    with pytest.raises(ValueError, match="outside the label set"):
        compute_metrics(["a"], ["zzz"], LABELS, INVALID)
    with pytest.raises(ValueError, match="same length"):
        compute_metrics(["a", "b"], ["a"], LABELS, INVALID)
    with pytest.raises(ValueError, match="no predictions"):
        compute_metrics([], [], LABELS, INVALID)
    with pytest.raises(ValueError, match="true labels"):
        compute_metrics(["zzz"], ["a"], LABELS, INVALID)
