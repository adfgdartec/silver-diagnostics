"""Deep, framework-neutral analysis for classification confusion matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def _validate_matrix(matrix: Sequence[Sequence[int]], labels: Sequence[str]) -> Tuple[Tuple[int, ...], ...]:
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("confusion matrix must be a non-empty square matrix")
    if len(labels) != len(matrix) or len(set(labels)) != len(labels):
        raise ValueError("labels must match the matrix dimensions and be unique")
    normalized = []
    for row in matrix:
        values = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("confusion matrix values must be non-negative integers")
            values.append(value)
        normalized.append(tuple(values))
    if sum(sum(row) for row in normalized) == 0:
        raise ValueError("confusion matrix must contain at least one sample")
    return tuple(normalized)


def _safe(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class ConfusionMatrixReport:
    labels: Tuple[str, ...]
    matrix: Tuple[Tuple[int, ...], ...]
    row_normalized: Tuple[Tuple[float, ...], ...]
    column_normalized: Tuple[Tuple[float, ...], ...]
    per_class: Tuple[Dict[str, Any], ...]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    top_confusions: Tuple[Dict[str, Any], ...]
    group_labels: Tuple[str, ...] = ()
    group_matrix: Tuple[Tuple[int, ...], ...] = ()
    within_group_error: int = 0
    cross_group_error: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "labels": list(self.labels),
            "matrix": [list(row) for row in self.matrix],
            "row_normalized": [list(row) for row in self.row_normalized],
            "column_normalized": [list(row) for row in self.column_normalized],
            "per_class": [dict(item) for item in self.per_class],
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "top_confusions": [dict(item) for item in self.top_confusions],
            "group_labels": list(self.group_labels),
            "group_matrix": [list(row) for row in self.group_matrix],
            "within_group_error": self.within_group_error,
            "cross_group_error": self.cross_group_error,
        }


def analyze_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    labels: Optional[Sequence[str]] = None,
    *,
    hierarchy: Optional[Mapping[str, str]] = None,
    top_k: int = 10,
) -> ConfusionMatrixReport:
    """Analyze class-level and hierarchy-level confusion in one pass.

    Rows represent actual classes and columns represent predicted classes.
    ``hierarchy`` optionally maps each class to a parent group, allowing the
    report to separate cross-family errors from within-family errors.
    """
    names = tuple(labels or (str(index) for index in range(len(matrix))))
    values = _validate_matrix(matrix, names)
    size = len(values)
    row_totals = tuple(sum(row) for row in values)
    column_totals = tuple(sum(values[row][column] for row in range(size)) for column in range(size))
    total = sum(row_totals)
    row_normalized = tuple(tuple(_safe(value, row_totals[row]) for value in values[row]) for row in range(size))
    column_normalized = tuple(tuple(_safe(values[row][column], column_totals[column]) for column in range(size)) for row in range(size))
    per_class = []
    for index, label in enumerate(names):
        true_positive = values[index][index]
        false_positive = column_totals[index] - true_positive
        false_negative = row_totals[index] - true_positive
        precision = _safe(true_positive, true_positive + false_positive)
        recall = _safe(true_positive, true_positive + false_negative)
        f1 = _safe(2 * precision * recall, precision + recall)
        true_negative = total - true_positive - false_positive - false_negative
        per_class.append({
            "label": label, "true_positive": true_positive,
            "false_positive": false_positive, "false_negative": false_negative,
            "true_negative": true_negative, "support": row_totals[index],
            "predicted_support": column_totals[index], "precision": precision,
            "recall": recall, "f1": f1, "specificity": _safe(true_negative, true_negative + false_positive),
            "error_rate": _safe(false_negative, row_totals[index]),
        })
    top = []
    for actual in range(size):
        for predicted in range(size):
            if actual != predicted and values[actual][predicted]:
                top.append({"actual": names[actual], "predicted": names[predicted], "count": values[actual][predicted],
                            "rate": _safe(values[actual][predicted], row_totals[actual])})
    top.sort(key=lambda item: (-item["count"], item["actual"], item["predicted"]))
    group_names: Tuple[str, ...] = ()
    group_matrix: Tuple[Tuple[int, ...], ...] = ()
    within = cross = 0
    if hierarchy is not None:
        missing = [label for label in names if label not in hierarchy]
        if missing:
            raise ValueError("hierarchy is missing labels: %s" % ", ".join(missing))
        group_names = tuple(sorted(set(hierarchy.values())))
        positions = {group: index for index, group in enumerate(group_names)}
        grouped = [[0 for _ in group_names] for _ in group_names]
        for actual in range(size):
            for predicted in range(size):
                grouped[positions[hierarchy[names[actual]]]][positions[hierarchy[names[predicted]]]] += values[actual][predicted]
                if actual != predicted:
                    if hierarchy[names[actual]] == hierarchy[names[predicted]]:
                        within += values[actual][predicted]
                    else:
                        cross += values[actual][predicted]
        group_matrix = tuple(tuple(row) for row in grouped)
    precisions = [item["precision"] for item in per_class]
    recalls = [item["recall"] for item in per_class]
    f1s = [item["f1"] for item in per_class]
    return ConfusionMatrixReport(
        labels=names, matrix=values, row_normalized=row_normalized, column_normalized=column_normalized,
        per_class=tuple(per_class), accuracy=_safe(sum(values[i][i] for i in range(size)), total),
        macro_precision=sum(precisions) / size, macro_recall=sum(recalls) / size,
        macro_f1=sum(f1s) / size, weighted_f1=_safe(sum(f1s[i] * row_totals[i] for i in range(size)), total),
        balanced_accuracy=sum(recalls) / size, top_confusions=tuple(top[:max(0, top_k)]),
        group_labels=group_names, group_matrix=group_matrix,
        within_group_error=within, cross_group_error=cross,
    )


@dataclass(frozen=True)
class HierarchicalConfusionReport:
    levels: Mapping[str, ConfusionMatrixReport]
    accuracy_by_level: Mapping[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {"accuracy_by_level": dict(self.accuracy_by_level),
                "levels": {name: report.to_dict() for name, report in self.levels.items()}}


def analyze_hierarchical_confusion(
    levels: Mapping[str, Sequence[Sequence[int]]],
    labels: Mapping[str, Sequence[str]],
) -> HierarchicalConfusionReport:
    """Analyze coarse-to-fine confusion matrices and compare level accuracy."""
    if not levels:
        raise ValueError("at least one classification level is required")
    missing = sorted(set(levels) - set(labels))
    if missing:
        raise ValueError("labels are missing levels: %s" % ", ".join(missing))
    reports = {name: analyze_confusion_matrix(matrix, labels[name]) for name, matrix in levels.items()}
    return HierarchicalConfusionReport(reports, {name: report.accuracy for name, report in reports.items()})
