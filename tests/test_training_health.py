import pytest

from silver_diagnostics import diagnose_training_health


def test_healthy_training_reports_best_step_and_markdown():
    report = diagnose_training_health([
        {"loss": 1.0}, {"loss": 0.7}, {"loss": 0.4}, {"loss": 0.2},
    ], patience=3)

    assert report.healthy
    assert report.score == 100
    assert report.best_step == 3
    assert report.best_value == 0.2
    assert "Training health" in report.to_markdown()
    assert report.to_dict()["schema"] == "silver.diagnostics/training-health-1"


def test_plateau_and_overfit_produce_ranked_actions():
    report = diagnose_training_health([
        {"loss": 1.0, "val_loss": 1.0},
        {"loss": 0.8, "val_loss": 0.8},
        {"loss": 0.6, "val_loss": 0.9},
        {"loss": 0.5, "val_loss": 1.0},
        {"loss": 0.5, "val_loss": 1.1},
    ], patience=2)

    codes = {item.code for item in report.diagnostics}
    assert {"metric_plateau", "probable_overfit"} <= codes
    assert report.status == "warning"
    assert report.recommendations[0].priority == "high"


def test_divergence_is_critical_and_actionable():
    report = diagnose_training_health([
        {"loss": 1.0}, {"loss": 0.2}, {"loss": 2.0},
    ], patience=2)

    assert report.status == "critical"
    assert any(item.code == "metric_divergence" for item in report.diagnostics)
    assert any(item.code == "restore_best_checkpoint" for item in report.recommendations)
    with pytest.raises(RuntimeError, match="critical"):
        report.raise_if_critical()


def test_max_mode_selects_highest_metric():
    report = diagnose_training_health([
        {"accuracy": 0.3}, {"accuracy": 0.8}, {"accuracy": 0.7},
    ], primary_metric="accuracy", mode="max", patience=2)
    assert report.best_step == 1
    assert report.best_value == 0.8


@pytest.mark.parametrize("kwargs", [
    {"mode": "sideways"}, {"patience": 1}, {"min_delta": -1},
])
def test_invalid_health_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        diagnose_training_health([{"loss": 1.0}], **kwargs)
