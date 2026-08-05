from typing import Any, List

from .models import Diagnostic, DiagnosticReport
from .utils import is_finite


def diagnose_dataset(
    features: List[List[float]], labels: List[float]
) -> DiagnosticReport:
    diagnostics: List[Diagnostic] = []
    if len(features) == 0:
        diagnostics.append(
            Diagnostic(
                code="empty_dataset",
                severity="error",
                message="The dataset contains no feature rows.",
                details={},
            )
        )
    if len(features) != len(labels):
        diagnostics.append(
            Diagnostic(
                code="length_mismatch",
                severity="error",
                message="Features and labels must contain the same number of rows.",
                details={"features": len(features), "labels": len(labels)},
            )
        )
    if features:
        expected_width = len(features[0])
        inconsistent_widths = [
            (index, len(row))
            for index, row in enumerate(features)
            if len(row) != expected_width
        ]
        if inconsistent_widths:
            diagnostics.append(
                Diagnostic(
                    code="feature_width_mismatch",
                    severity="error",
                    message="Feature rows do not have a consistent width.",
                    details={
                        "expected": expected_width,
                        "mismatches": inconsistent_widths[:5],
                    },
                )
            )
        for row_index, row in enumerate(features):
            for column_index, value in enumerate(row):
                if not isinstance(value, (int, float)) or not is_finite(value):
                    diagnostics.append(
                        Diagnostic(
                            code="non_finite_feature",
                            severity="error",
                            message="A feature is not finite.",
                            details={
                                "row": row_index,
                                "column": column_index,
                                "value": value,
                            },
                        )
                    )
    for label_index, label in enumerate(labels):
        if not isinstance(label, (int, float)) or not is_finite(label):
            diagnostics.append(
                Diagnostic(
                    code="non_finite_label",
                    severity="error",
                    message="A label is not finite.",
                    details={"index": label_index, "value": label},
                )
            )
    return DiagnosticReport(
        valid=not any(diagnostic.severity == "error" for diagnostic in diagnostics),
        diagnostics=diagnostics,
    )
