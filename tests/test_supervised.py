import pytest

from silver_diagnostics import assess_supervised_system


def test_transformer_assessment_reports_requirements_and_visuals():
    assessment = assess_supervised_system("transformer", "classification", {
        "labels": True, "finite_features": True, "train_validation_test": True,
        "sequence_order": True, "scaled_numeric": True,
    })
    assert assessment.valid
    assert "attention_context" in {item.key for item in assessment.principles}
    assert "confusion_matrix" in assessment.recommended_visualizations


def test_cnn_assessment_blocks_without_spatial_layout():
    assessment = assess_supervised_system("cnn", "classification", {
        "labels": True, "finite_features": True, "train_validation_test": True,
    })
    assert not assessment.valid
    assert any("spatial_layout" in item for item in assessment.blockers)


def test_unknown_family_is_explicit():
    with pytest.raises(ValueError, match="unknown algorithm family"):
        assess_supervised_system("invented", "classification", {})
