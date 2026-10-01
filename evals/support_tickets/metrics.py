"""Classification metrics via scikit-learn, with sample sizes attached to every number."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


class PerLabel(TypedDict):
    label: str
    precision: float
    recall: float
    f1: float
    support: int  # true tickets with this label


class Averages(TypedDict):
    precision: float
    recall: float
    f1: float


class Metrics(TypedDict):
    n: int
    correct: int
    accuracy: float
    invalid_predictions: int
    macro: Averages
    weighted: Averages
    per_label: list[PerLabel]
    confusion_labels: list[str]  # rows = true label, columns = predicted label
    confusion_matrix: list[list[int]]


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str], invalid: str) -> Metrics:
    """Per-label precision/recall/F1 over `labels`; predictions outside `labels` must be `invalid`.

    `invalid` predictions count as wrong. They appear as an extra confusion-matrix column
    (never as a row, since no true label is invalid) and lower recall for the true label.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("no predictions to score")
    known = set(labels)
    bad = sorted({p for p in y_pred if p not in known and p != invalid})
    if bad:
        raise ValueError(f"predictions outside the label set: {bad}")
    if any(t not in known for t in y_true):
        raise ValueError("true labels must all be in the label set")

    lab = list(labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=lab, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=lab, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=lab, average="weighted", zero_division=0)
    cm_labels = [*lab, invalid]
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
    correct = sum(1 for t, q in zip(y_true, y_pred, strict=True) if t == q)
    return {
        "n": len(y_true),
        "correct": correct,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "invalid_predictions": sum(1 for q in y_pred if q == invalid),
        "macro": {
            "precision": round(float(macro[0]), 4),
            "recall": round(float(macro[1]), 4),
            "f1": round(float(macro[2]), 4),
        },
        "weighted": {
            "precision": round(float(weighted[0]), 4),
            "recall": round(float(weighted[1]), 4),
            "f1": round(float(weighted[2]), 4),
        },
        "per_label": [
            {
                "label": lab[i],
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1": round(float(f[i]), 4),
                "support": int(s[i]),
            }
            for i in range(len(lab))
        ],
        "confusion_labels": cm_labels,
        # Drop the all-zero "(invalid)" row: no ticket's true label is invalid.
        "confusion_matrix": [[int(v) for v in row] for row in cm[: len(lab)]],
    }
