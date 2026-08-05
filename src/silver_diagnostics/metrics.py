from typing import Dict

from .models import Diagnostic, DiagnosticReport
from .utils import is_finite


def diagnose_metrics(metrics: Dict[str, float]) -> DiagnosticReport:
    diagnostics = []
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
            and abs(value) > 1e4
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
