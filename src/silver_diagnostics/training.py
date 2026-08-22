"""Actionable health analysis for complete training histories."""

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .metrics import diagnose_metric_history
from .models import Diagnostic


@dataclass(frozen=True)
class Recommendation:
    code: str
    title: str
    action: str
    priority: str = "medium"

    def to_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "title": self.title,
            "action": self.action,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class TrainingHealthReport:
    status: str
    score: int
    primary_metric: str
    best_step: Optional[int]
    best_value: Optional[float]
    diagnostics: Tuple[Diagnostic, ...]
    recommendations: Tuple[Recommendation, ...]

    @property
    def healthy(self) -> bool:
        return self.status == "healthy"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "silver.diagnostics/training-health-1",
            "status": self.status,
            "score": self.score,
            "primary_metric": self.primary_metric,
            "best_step": self.best_step,
            "best_value": self.best_value,
            "diagnostics": [
                {
                    "code": item.code,
                    "severity": item.severity,
                    "message": item.message,
                    "details": dict(item.details),
                }
                for item in self.diagnostics
            ],
            "recommendations": [item.to_dict() for item in self.recommendations],
        }

    def to_markdown(self) -> str:
        icon = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}[self.status]
        lines = [
            "# Training health",
            "",
            "%s **%s** — score **%d/100**" % (icon, self.status.upper(), self.score),
            "",
        ]
        if self.best_step is not None:
            lines.append(
                "Best `%s`: `%s` at step `%d`." % (
                    self.primary_metric, self.best_value, self.best_step,
                )
            )
            lines.append("")
        if self.diagnostics:
            lines.extend(["## Findings", ""])
            lines.extend(
                "- **%s** `%s`: %s" % (
                    item.severity.upper(), item.code, item.message,
                )
                for item in self.diagnostics
            )
            lines.append("")
        if self.recommendations:
            lines.extend(["## Next actions", ""])
            lines.extend(
                "1. **%s** — %s" % (item.title, item.action)
                for item in self.recommendations
            )
        return "\n".join(lines).rstrip() + "\n"

    def raise_if_critical(self) -> "TrainingHealthReport":
        if self.status == "critical":
            raise RuntimeError("training health is critical: %s" % ", ".join(
                item.code for item in self.diagnostics if item.severity == "error"
            ))
        return self


def diagnose_training_health(
    history: Sequence[Mapping[str, float]],
    *,
    primary_metric: str = "loss",
    mode: str = "min",
    patience: int = 5,
    min_delta: float = 0.0,
    validation_metric: Optional[str] = None,
) -> TrainingHealthReport:
    """Turn raw metric history into a ranked, actionable training assessment."""
    if mode not in ("min", "max"):
        raise ValueError("mode must be 'min' or 'max'")
    if patience < 2:
        raise ValueError("patience must be at least 2")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")
    base = diagnose_metric_history(history, required=(primary_metric,))
    diagnostics = list(base.diagnostics)
    values = [snapshot.get(primary_metric) for snapshot in history]
    valid_values = all(_finite_number(value) for value in values)
    best_step: Optional[int] = None
    best_value: Optional[float] = None
    if valid_values and values:
        numeric = [float(value) for value in values]
        best_step = min(range(len(numeric)), key=numeric.__getitem__) if mode == "min" \
            else max(range(len(numeric)), key=numeric.__getitem__)
        best_value = numeric[best_step]
        window = numeric[-patience:]
        if len(window) == patience and _improvement(window, mode) <= min_delta:
            diagnostics.append(Diagnostic(
                "metric_plateau", "warning",
                "%s has plateaued over the last %d steps." % (primary_metric, patience),
                {"metric": primary_metric, "patience": patience, "window": window},
            ))
        if len(numeric) >= 3 and _substantially_worse(
            numeric[-1], best_value, numeric[0], mode
        ):
            diagnostics.append(Diagnostic(
                "metric_divergence", "error",
                "%s has diverged substantially from its best value." % primary_metric,
                {"metric": primary_metric, "best": best_value, "latest": numeric[-1]},
            ))
    validation_name = validation_metric or _validation_metric(history)
    if validation_name:
        _append_overfit_diagnostic(diagnostics, history, primary_metric, validation_name, mode)
    recommendations = _recommendations(diagnostics)
    errors = sum(item.severity == "error" for item in diagnostics)
    warnings = sum(item.severity == "warning" for item in diagnostics)
    score = max(0, 100 - errors * 35 - warnings * 12)
    status = "critical" if errors else "warning" if warnings else "healthy"
    return TrainingHealthReport(
        status, score, primary_metric, best_step, best_value,
        tuple(diagnostics), recommendations,
    )


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) \
        and math.isfinite(float(value))


def _improvement(values: Sequence[float], mode: str) -> float:
    if mode == "min":
        return values[0] - min(values[1:])
    return max(values[1:]) - values[0]


def _substantially_worse(latest: float, best: float, initial: float, mode: str) -> bool:
    scale = max(abs(best), 1e-12)
    if mode == "min":
        return latest > initial and latest > best and (latest - best) / scale >= 0.5
    return latest < initial and latest < best and (best - latest) / scale >= 0.5


def _validation_metric(history: Sequence[Mapping[str, float]]) -> Optional[str]:
    if not history:
        return None
    for name in ("val_loss", "validation_loss", "val_accuracy", "validation_accuracy"):
        if name in history[0]:
            return name
    return None


def _append_overfit_diagnostic(
    diagnostics: list,
    history: Sequence[Mapping[str, float]],
    training_name: str,
    validation_name: str,
    mode: str,
) -> None:
    if len(history) < 3:
        return
    train = [item.get(training_name) for item in history]
    validation = [item.get(validation_name) for item in history]
    if not all(_finite_number(item) for item in train + validation):
        return
    train_improved = float(train[-1]) < float(train[0]) if mode == "min" \
        else float(train[-1]) > float(train[0])
    validation_worsened = float(validation[-1]) > min(float(v) for v in validation) \
        if mode == "min" else float(validation[-1]) < max(float(v) for v in validation)
    if train_improved and validation_worsened:
        diagnostics.append(Diagnostic(
            "probable_overfit", "warning",
            "Training improved while %s moved away from its best value." % validation_name,
            {"training_metric": training_name, "validation_metric": validation_name},
        ))


def _recommendations(diagnostics: Sequence[Diagnostic]) -> Tuple[Recommendation, ...]:
    catalog = {
        "non_finite_metric": Recommendation(
            "stabilize_numeric", "Stop and stabilize numerics",
            "Check input scaling, reduce the learning rate, and enable gradient clipping.",
            "high",
        ),
        "missing_metric": Recommendation(
            "restore_observability", "Restore metric logging",
            "Emit the required primary metric at every evaluation step.", "high",
        ),
        "metric_divergence": Recommendation(
            "restore_best_checkpoint", "Restore the best checkpoint",
            "Roll back to the reported best step, lower the learning rate, and resume cautiously.",
            "high",
        ),
        "metric_plateau": Recommendation(
            "adjust_plateau", "Respond to the plateau",
            "Reduce the learning rate or stop early if validation quality is no longer improving.",
        ),
        "probable_overfit": Recommendation(
            "reduce_overfit", "Reduce overfitting",
            "Use the best validation checkpoint and add regularization, augmentation, or more data.",
            "high",
        ),
        "loss_regression": Recommendation(
            "inspect_regression", "Inspect the regression",
            "Compare recent batches and configuration changes with the best-performing window.",
        ),
        "exploding_gradient": Recommendation(
            "clip_gradients", "Clip gradients",
            "Enable gradient clipping and inspect activation and input scales.", "high",
        ),
    }
    selected = []
    seen = set()
    for diagnostic in diagnostics:
        recommendation = catalog.get(diagnostic.code)
        if recommendation and recommendation.code not in seen:
            selected.append(recommendation)
            seen.add(recommendation.code)
    priority = {"high": 0, "medium": 1, "low": 2}
    return tuple(sorted(selected, key=lambda item: (priority[item.priority], item.code)))
