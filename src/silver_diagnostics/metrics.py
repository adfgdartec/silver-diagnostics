from typing import Dict, Iterable, Mapping, Sequence

from .models import Diagnostic, DiagnosticReport
from .utils import is_finite


def diagnose_metrics(
    metrics: Dict[str, float], *, gradient_threshold: float = 1e4,
    required: Iterable[str] = (),
) -> DiagnosticReport:
    diagnostics = []
    for name in required:
        if name not in metrics:
            diagnostics.append(Diagnostic("missing_metric", "error",
                f"Required metric {name} is missing.", {"name": name}))
    for name, value in metrics.items():
        if not isinstance(value, (int, float)) or not is_finite(value):
            diagnostics.append(
                Diagnostic(
                    code="non_finite_metric",
                    severity="error",
                    message=f"{name} is not finite.",
                    details={"name": name, "value": value},
                )
            )
        if (
            "gradient" in name.lower()
            and isinstance(value, (int, float))
            and is_finite(value)
            and abs(value) > gradient_threshold
        ):
            diagnostics.append(
                Diagnostic(
                    code="exploding_gradient",
                    severity="warning",
                    message=f"{name} is unusually large and may indicate exploding gradients.",
                    details={"name": name, "value": value},
                )
            )
    return DiagnosticReport(
        valid=not any(diagnostic.severity == "error" for diagnostic in diagnostics),
        diagnostics=diagnostics,
    )


def diagnose_metric_history(
    history: Sequence[Mapping[str, float]], *, required: Iterable[str] = (),
) -> DiagnosticReport:
    """Diagnose a sequence of metric snapshots and flag regressions in loss."""
    diagnostics = []
    if not history:
        return DiagnosticReport(False, [Diagnostic("empty_history", "error",
            "Metric history is empty.", {})])
    for index, metrics in enumerate(history):
        report = diagnose_metrics(dict(metrics), required=required)
        diagnostics.extend(Diagnostic(item.code, item.severity,
            f"step {index}: {item.message}", {"step": index, **item.details})
            for item in report.diagnostics)
    losses = [item.get("loss") for item in history]
    if all(isinstance(value, (int, float)) and is_finite(value) for value in losses):
        if len(losses) >= 3 and losses[-1] > losses[0] * 1.25:
            diagnostics.append(Diagnostic("loss_regression", "warning",
                "Loss increased substantially over the recorded history.",
                {"first": losses[0], "last": losses[-1]}))
    return DiagnosticReport(not any(item.severity == "error" for item in diagnostics), diagnostics)
