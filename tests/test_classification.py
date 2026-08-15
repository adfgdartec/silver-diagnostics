import pytest

from silver_diagnostics import analyze_confusion_matrix, analyze_hierarchical_confusion


def test_confusion_matrix_has_deep_per_class_and_normalized_analysis():
    report = analyze_confusion_matrix(
        [[8, 2, 0], [1, 6, 1], [0, 2, 10]],
        ["cat", "dog", "bird"],
        hierarchy={"cat": "animal", "dog": "animal", "bird": "animal"},
    )
    dog = next(item for item in report.per_class if item["label"] == "dog")
    assert report.accuracy == pytest.approx(24 / 30)
    assert dog["false_positive"] == 4
    assert dog["recall"] == pytest.approx(6 / 8)
    assert report.row_normalized[0][1] == pytest.approx(0.2)
    assert report.top_confusions[0]["count"] == 2
    assert report.within_group_error == 6
    assert report.cross_group_error == 0
    assert report.to_dict()["group_labels"] == ["animal"]


def test_hierarchical_confusion_compares_levels():
    report = analyze_hierarchical_confusion(
        {"coarse": [[8, 2], [1, 9]], "fine": [[7, 1, 0], [1, 7, 1], [0, 1, 8]]},
        {"coarse": ["healthy", "diseased"], "fine": ["a", "b", "c"]},
    )
    assert report.accuracy_by_level["coarse"] == pytest.approx(0.85)
    assert report.accuracy_by_level["fine"] == pytest.approx(22 / 26)


def test_confusion_matrix_rejects_invalid_values():
    with pytest.raises(ValueError, match="non-negative integers"):
        analyze_confusion_matrix([[1, -1], [0, 1]])
