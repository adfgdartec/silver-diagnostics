import pytest
import math
from silver_diagnostics import (
    Diagnostic,
    DiagnosticReport,
    diagnose_dataset,
    diagnose_metrics,
    diagnose_metric_history,
)


class TestDiagnostic:
    def test_diagnostic_creation(self):
        diagnostic = Diagnostic(
            code="test_code",
            severity="error",
            message="Test message",
            details={"key": "value"},
        )
        assert diagnostic.code == "test_code"
        assert diagnostic.severity == "error"
        assert diagnostic.message == "Test message"
        assert diagnostic.details == {"key": "value"}


class TestDiagnosticReport:
    def test_diagnostic_report_creation(self):
        diagnostics = [
            Diagnostic(code="test", severity="error", message="Test", details={})
        ]
        report = DiagnosticReport(valid=False, diagnostics=diagnostics)
        assert report.valid == False
        assert len(report.diagnostics) == 1
        assert len(report.errors) == 1
        assert report.summary() == {"error": 1, "warning": 0, "info": 0}
        with pytest.raises(ValueError, match="Test"):
            report.raise_if_invalid()

    def test_report_serializes(self):
        report = diagnose_metrics({"loss": 0.5})
        assert report.to_dict() == {"valid": True, "diagnostics": []}


class TestDiagnoseDataset:
    def test_empty_dataset(self):
        features = []
        labels = []
        report = diagnose_dataset(features, labels)

        assert report.valid == False
        assert any(d.code == "empty_dataset" for d in report.diagnostics)

    def test_length_mismatch(self):
        features = [[1.0, 2.0], [3.0, 4.0]]
        labels = [0.0]  # Only one label
        report = diagnose_dataset(features, labels)

        assert report.valid == False
        assert any(d.code == "length_mismatch" for d in report.diagnostics)

    def test_feature_width_mismatch(self):
        features = [[1.0, 2.0], [3.0]]  # Different widths
        labels = [0.0, 1.0]
        report = diagnose_dataset(features, labels)

        assert report.valid == False
        assert any(d.code == "feature_width_mismatch" for d in report.diagnostics)

    def test_non_finite_feature(self):
        features = [[1.0, float("nan")], [3.0, 4.0]]
        labels = [0.0, 1.0]
        report = diagnose_dataset(features, labels)

        assert report.valid == False
        assert any(d.code == "non_finite_feature" for d in report.diagnostics)

    def test_infinite_feature(self):
        features = [[1.0, float("inf")], [3.0, 4.0]]
        labels = [0.0, 1.0]
        report = diagnose_dataset(features, labels)

        assert report.valid == False
        assert any(d.code == "non_finite_feature" for d in report.diagnostics)

    def test_non_finite_label(self):
        features = [[1.0, 2.0], [3.0, 4.0]]
        labels = [0.0, float("nan")]
        report = diagnose_dataset(features, labels)

        assert report.valid == False
        assert any(d.code == "non_finite_label" for d in report.diagnostics)

    def test_string_feature(self):
        features = [[1.0, "invalid"], [3.0, 4.0]]
        labels = [0.0, 1.0]
        report = diagnose_dataset(features, labels)

        assert report.valid == False
        assert any(d.code == "non_finite_feature" for d in report.diagnostics)

    def test_valid_dataset(self):
        features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        labels = [0.0, 1.0, 0.0]
        report = diagnose_dataset(features, labels)

        assert report.valid == True
        assert len(report.diagnostics) == 0

    def test_single_feature_vector(self):
        features = [[1.0, 2.0, 3.0]]
        labels = [0.0]
        report = diagnose_dataset(features, labels)

        assert report.valid == True

    def test_consistent_feature_width(self):
        features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        labels = [0.0, 1.0, 0.0]
        report = diagnose_dataset(features, labels)

        # Should not have width mismatch error
        assert not any(d.code == "feature_width_mismatch" for d in report.diagnostics)


class TestDiagnoseMetrics:
    def test_metric_history_required_and_regression(self):
        report = diagnose_metric_history([{"loss": 1.0}, {"loss": 1.3}, {"loss": 1.5}], required=["accuracy"])
        assert not report.valid
        assert any(item.code == "missing_metric" for item in report.diagnostics)
        assert any(item.code == "loss_regression" for item in report.diagnostics)

    def test_non_finite_metric(self):
        metrics = {"loss": float("nan"), "accuracy": 0.9}
        report = diagnose_metrics(metrics)

        assert report.valid == False
        assert any(d.code == "non_finite_metric" for d in report.diagnostics)

    def test_infinite_metric(self):
        metrics = {"loss": float("inf"), "accuracy": 0.9}
        report = diagnose_metrics(metrics)

        assert report.valid == False
        assert any(d.code == "non_finite_metric" for d in report.diagnostics)

    def test_string_metric(self):
        metrics = {"loss": "invalid", "accuracy": 0.9}
        report = diagnose_metrics(metrics)

        assert report.valid == False
        assert any(d.code == "non_finite_metric" for d in report.diagnostics)

    def test_exploding_gradient_warning(self):
        metrics = {"gradient_norm": 1e5, "loss": 0.5}
        report = diagnose_metrics(metrics)

        # Should have warning but still be valid (no errors)
        assert report.valid == True
        assert any(d.code == "exploding_gradient" for d in report.diagnostics)
        gradient_warnings = [
            d for d in report.diagnostics if d.code == "exploding_gradient"
        ]
        assert len(gradient_warnings) == 1

    def test_exploding_gradient_small_value_no_warning(self):
        metrics = {"gradient_norm": 100.0, "loss": 0.5}
        report = diagnose_metrics(metrics)

        # Should not have exploding gradient warning
        assert not any(d.code == "exploding_gradient" for d in report.diagnostics)

    def test_exploding_gradient_case_insensitive(self):
        metrics = {"Gradient": 1e5, "loss": 0.5}
        report = diagnose_metrics(metrics)

        assert any(d.code == "exploding_gradient" for d in report.diagnostics)

    def test_valid_metrics(self):
        metrics = {"loss": 0.5, "accuracy": 0.9, "f1": 0.85}
        report = diagnose_metrics(metrics)

        assert report.valid == True
        assert len(report.diagnostics) == 0

    def test_negative_metrics_valid(self):
        metrics = {"loss": -0.5, "gradient": -100.0}
        report = diagnose_metrics(metrics)

        assert report.valid == True

    def test_zero_metrics_valid(self):
        metrics = {"loss": 0.0, "accuracy": 0.0}
        report = diagnose_metrics(metrics)

        assert report.valid == True

    def test_multiple_metrics_multiple_errors(self):
        metrics = {"loss": float("nan"), "accuracy": float("inf"), "gradient_norm": 1e6}
        report = diagnose_metrics(metrics)

        assert report.valid == False
        error_count = sum(1 for d in report.diagnostics if d.severity == "error")
        assert error_count == 2

        warning_count = sum(1 for d in report.diagnostics if d.severity == "warning")
        assert warning_count == 1


class TestDiagnosticDetails:
    def test_empty_dataset_details(self):
        features = []
        labels = []
        report = diagnose_dataset(features, labels)

        empty_diag = next(d for d in report.diagnostics if d.code == "empty_dataset")
        assert empty_diag.details == {}

    def test_length_mismatch_details(self):
        features = [[1.0], [2.0], [3.0]]
        labels = [0.0, 1.0]
        report = diagnose_dataset(features, labels)

        mismatch_diag = next(
            d for d in report.diagnostics if d.code == "length_mismatch"
        )
        assert mismatch_diag.details["features"] == 3
        assert mismatch_diag.details["labels"] == 2

    def test_feature_width_mismatch_details(self):
        features = [[1.0, 2.0], [3.0]]
        labels = [0.0, 1.0]
        report = diagnose_dataset(features, labels)

        width_diag = next(
            d for d in report.diagnostics if d.code == "feature_width_mismatch"
        )
        assert "expected" in width_diag.details
        assert "mismatches" in width_diag.details

    def test_non_finite_feature_details(self):
        features = [[1.0, float("nan")]]
        labels = [0.0]
        report = diagnose_dataset(features, labels)

        feature_diag = next(
            d for d in report.diagnostics if d.code == "non_finite_feature"
        )
        assert "row" in feature_diag.details
        assert "column" in feature_diag.details
        assert "value" in feature_diag.details

    def test_exploding_gradient_details(self):
        metrics = {"gradient_norm": 1e5}
        report = diagnose_metrics(metrics)

        gradient_diag = next(
            d for d in report.diagnostics if d.code == "exploding_gradient"
        )
        assert gradient_diag.details["name"] == "gradient_norm"
        assert gradient_diag.details["value"] == 1e5


class TestEdgeCases:
    def test_very_large_numbers(self):
        features = [[1e308, 2e307]]
        labels = [0.0]
        report = diagnose_dataset(features, labels)

        # Very large numbers are still finite
        assert report.valid == True

    def test_very_small_numbers(self):
        features = [[1e-308, 2e-308]]
        labels = [0.0]
        report = diagnose_dataset(features, labels)

        # Very small numbers are still finite
        assert report.valid == True

    def test_negative_infinity(self):
        features = [[1.0, float("-inf")]]
        labels = [0.0]
        report = diagnose_dataset(features, labels)

        assert report.valid == False
        assert any(d.code == "non_finite_feature" for d in report.diagnostics)

    def test_zero_denominator_like_scenario(self):
        # Test a scenario that might cause division by zero in some implementations
        features = [[1.0, 0.0], [2.0, 0.0]]
        labels = [0.0, 1.0]
        report = diagnose_dataset(features, labels)

        assert report.valid == True
