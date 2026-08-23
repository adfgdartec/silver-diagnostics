"""Framework-neutral comparison of two Silver experiment reports."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ReportComparison:
    verdict: str
    score_delta: float | None
    loss_delta: float | None
    health_delta: int | None
    resolved_decisions: tuple[str, ...]
    new_decisions: tuple[str, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "silver.diagnostics/report-comparison-1",
            "verdict": self.verdict,
            "score_delta": self.score_delta,
            "loss_delta": self.loss_delta,
            "health_delta": self.health_delta,
            "resolved_decisions": list(self.resolved_decisions),
            "new_decisions": list(self.new_decisions),
            "summary": self.summary,
        }


def compare_reports(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> ReportComparison:
    """Compare reports and explain whether the candidate is safer to keep."""
    old_plan = baseline.get("decision_plan", {})
    new_plan = candidate.get("decision_plan", {})
    old_codes = {str(item.get("code")) for item in old_plan.get("decisions", ()) if isinstance(item, Mapping)}
    new_codes = {str(item.get("code")) for item in new_plan.get("decisions", ()) if isinstance(item, Mapping)}
    resolved = tuple(sorted(old_codes - new_codes))
    added = tuple(sorted(new_codes - old_codes))
    old_health = _number(baseline.get("health", {}).get("score"))
    new_health = _number(candidate.get("health", {}).get("score"))
    old_loss = _number(baseline.get("training", {}).get("best_loss"))
    new_loss = _number(candidate.get("training", {}).get("best_loss"))
    score_delta = _delta(old_plan.get("score"), new_plan.get("score"))
    health_delta = int(new_health - old_health) if old_health is not None and new_health is not None else None
    loss_delta = _delta(old_loss, new_loss)
    improved = (health_delta is not None and health_delta > 0) or (loss_delta is not None and loss_delta < 0)
    regressed = (health_delta is not None and health_delta < 0) or (loss_delta is not None and loss_delta > 0)
    verdict = "improved" if improved and not regressed else "regressed" if regressed and not improved else "mixed" if improved or regressed else "unchanged"
    summary = {
        "improved": "Keep the candidate: measurable health or loss improvement is supported by the evidence.",
        "regressed": "Reject the candidate: the measured outcome worsened; restore the baseline before the next change.",
        "mixed": "Inspect the new decision cards: one signal improved while another regressed.",
        "unchanged": "Treat the candidate as inconclusive and change one controlled variable next.",
    }[verdict]
    return ReportComparison(verdict, score_delta, loss_delta, health_delta, resolved, added, summary)


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(old: Any, new: Any) -> float | None:
    old_value, new_value = _number(old), _number(new)
    return new_value - old_value if old_value is not None and new_value is not None else None
