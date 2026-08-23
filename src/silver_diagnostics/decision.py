"""Evidence-backed debugging decisions for complete ML experiments."""

from dataclasses import asdict, dataclass
from html import escape
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DecisionStep:
    code: str
    domain: str
    priority: str
    title: str
    evidence: Tuple[str, ...]
    action: str
    expected_effect: str
    verify: str

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["evidence"] = list(self.evidence)
        return value


@dataclass(frozen=True)
class DecisionPlan:
    status: str
    score: int
    decisions: Tuple[DecisionStep, ...]
    next_experiment: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "silver.diagnostics/decision-plan-1",
            "status": self.status,
            "score": self.score,
            "next_experiment": self.next_experiment,
            "decisions": [item.to_dict() for item in self.decisions],
        }

    def to_svg(self, *, title: str = "Debugging decision guide") -> str:
        return decision_plan_svg(self, title=title)


def build_debug_plan(
    *,
    training: Optional[Mapping[str, Any]] = None,
    layers: Sequence[Mapping[str, Any]] = (),
    neural_inputs: Optional[Mapping[str, Any]] = None,
    topology: Optional[Mapping[str, Any]] = None,
) -> DecisionPlan:
    """Turn Silver evidence contracts into ranked interventions and checks.

    Inputs are plain mappings so the planner can consume evidence from any
    framework. It never changes a model or dataset; it proposes one controlled
    next experiment and a concrete verification signal.
    """
    training = training or {}
    decisions = []
    seen = set()

    def add(step: DecisionStep) -> None:
        if step.code not in seen:
            decisions.append(step)
            seen.add(step.code)

    diagnostics = training.get("diagnostics", ())
    diagnostic_codes = {
        str(item.get("code")) for item in diagnostics if isinstance(item, Mapping)
    }
    for code in diagnostic_codes:
        _training_decision(code, training, add)

    for index, layer in enumerate(layers):
        name = str(layer.get("name", "layer-%d" % index))
        zero = _number(layer.get("activation_zero_fraction", layer.get("zero_fraction")))
        std = _number(layer.get("activation_std"))
        gradient = _number(layer.get("gradient_rms"))
        if zero is not None and zero >= 0.95:
            add(DecisionStep(
                "dead_activation", "architecture", "high",
                "Revive the inactive layer",
                ("%s is %.1f%% zero" % (name, 100 * zero),),
                "Check the preceding scale and bias, lower the learning rate, and test a non-saturating activation or a smaller initialization.",
                "More units should carry non-zero signal without materially increasing validation loss.",
                "Re-run the same inspection batch and require zero fraction below 95% while validation loss improves or holds.",
            ))
        if std is not None and std <= 1e-10:
            add(DecisionStep(
                "collapsed_activation", "architecture", "high",
                "Restore activation variance",
                ("%s activation σ is %.3g" % (name, std),),
                "Inspect input scaling and normalization around this layer; compare a narrower layer and a residual or skip connection.",
                "Activation variance becomes non-zero and downstream gradients become measurable.",
                "Compare activation σ and gradient RMS before and after the single architecture change on the same batch.",
            ))
        if gradient is not None and gradient <= 1e-8:
            add(DecisionStep(
                "vanishing_gradient", "optimization", "high",
                "Increase gradient signal",
                ("%s gradient RMS is %.3g" % (name, gradient),),
                "Reduce unnecessary depth, add a residual path, inspect initialization, and try a learning-rate sweep around the current value.",
                "Earlier layers receive a stable non-zero gradient and validation quality improves without exploding updates.",
                "Track per-layer gradient RMS for three runs and reject changes that only improve training loss.",
            ))
        if gradient is not None and gradient >= 100:
            add(DecisionStep(
                "exploding_gradient", "optimization", "critical",
                "Contain exploding gradients",
                ("%s gradient RMS is %.3g" % (name, gradient),),
                "Enable gradient clipping, lower the learning rate, and inspect non-finite inputs before changing architecture.",
                "Gradient RMS returns below the clipping threshold and the metric history stops diverging.",
                "Require finite loss for every step and compare clipped versus unclipped gradient norms.",
            ))

    if neural_inputs:
        features = neural_inputs.get("features", ())
        for feature in features:
            if not isinstance(feature, Mapping):
                continue
            missing = _number(feature.get("missing_rate"))
            name = str(feature.get("name", "feature"))
            unique = _number(feature.get("unique_values"))
            if missing is not None and missing >= 0.2:
                add(DecisionStep(
                    "repair_missingness", "data", "high",
                    "Repair high-missingness inputs",
                    ("%s is %.1f%% missing" % (name, 100 * missing),),
                    "Audit the source and imputation policy; add an explicit missingness indicator before tuning the model.",
                    "Missingness falls or becomes predictive through an intentional indicator, with validation performance stable across splits.",
                    "Compare a missingness-aware run against the current baseline using the same split and seed.",
                ))
            if unique is not None and unique >= 32 and str(feature.get("role")) == "categorical":
                add(DecisionStep(
                    "high_cardinality", "data", "medium",
                    "Change the high-cardinality encoding",
                    ("%s has %d observed categories" % (name, int(unique)),),
                    "Benchmark an embedding or frequency encoding against the current vocabulary index; keep an unknown bucket for serving.",
                    "Validation quality improves without a large train/serve mismatch or parameter blow-up.",
                    "Log unseen-category rate and compare validation loss, memory, and latency for both encodings.",
                ))

    if topology and not topology.get("nodes"):
        add(DecisionStep(
            "inspect_topology", "architecture", "medium",
            "Make the architecture observable",
            ("topology contains no layer nodes",),
            "Add a topology adapter or explicit layer mapping before comparing architectures.",
            "The next report identifies every trainable stage and its parameter contribution.",
            "Require a non-empty network topology artifact before accepting a model comparison.",
        ))

    priority = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    decisions.sort(key=lambda item: (priority[item.priority], item.domain, item.code))
    score = int(training.get("score", 100)) if _number(training.get("score")) is not None else 100
    score = max(0, min(100, score - 8 * max(0, len(decisions) - 1)))
    status = "critical" if any(item.priority == "critical" for item in decisions) else \
        "warning" if decisions or training.get("status") == "warning" else "healthy"
    next_experiment = (
        "Change one high-priority variable, preserve the seed and split, then compare the verification signals above."
        if decisions else
        "Keep this run as the baseline; test one controlled architecture or data change and compare validation quality."
    )
    return DecisionPlan(status, score, tuple(decisions), next_experiment)


def decision_plan_svg(plan: DecisionPlan, *, title: str = "Debugging decision guide") -> str:
    width = 1220
    row_height = 178
    height = 170 + max(1, len(plan.decisions)) * row_height
    colors = {"critical": "#ff6b7d", "high": "#ffbd5c", "medium": "#7de3ff", "low": "#9b7dff"}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="{escape(title)}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Inter,ui-sans-serif,system-ui,sans-serif}"
        ".title{fill:#f5f7fa;font-size:27px;font-weight:700}.meta{fill:#9fb0c3;font-size:12px}"
        ".heading{fill:#eaf2fb;font-size:15px;font-weight:700}.body{fill:#b9c7d6;font-size:12px}"
        ".evidence{fill:#79e0ff;font-size:12px}.label{fill:#8295aa;font-size:10px;letter-spacing:.1em}</style>",
        f'<rect width="{width}" height="{height}" rx="22" fill="#0a1017"/>',
        f'<text class="title" x="34" y="44">{escape(title)}</text>',
        f'<text class="meta" x="34" y="69">{escape(plan.status)} · decision confidence {plan.score}/100 · '
        f'{len(plan.decisions)} evidence-backed interventions</text>',
        '<rect x="34" y="88" width="1150" height="52" rx="12" fill="#111b25" stroke="#294054"/>',
        '<text class="heading" x="52" y="111">NEXT CONTROLLED EXPERIMENT</text>',
        f'<text class="body" x="52" y="129">{escape(plan.next_experiment)}</text>',
    ]
    if not plan.decisions:
        parts.append('<text class="meta" x="52" y="210">No intervention required; preserve this run as the baseline.</text>')
    for index, decision in enumerate(plan.decisions):
        y = 160 + index * row_height
        color = colors.get(decision.priority, "#7de3ff")
        evidence = " · ".join(decision.evidence)
        parts.extend([
            f'<rect x="34" y="{y}" width="1150" height="154" rx="14" fill="#101923" stroke="{color}"/>',
            f'<rect x="34" y="{y}" width="8" height="154" rx="4" fill="{color}"/>',
            f'<text class="label" x="60" y="{y + 23}">{escape(decision.priority.upper())} · {escape(decision.domain.upper())}</text>',
            f'<text class="heading" x="60" y="{y + 47}">{escape(decision.title)}</text>',
            f'<text class="evidence" x="60" y="{y + 70}">EVIDENCE · {escape(evidence[:125])}</text>',
            f'<text class="body" x="60" y="{y + 94}">ACT · {escape(decision.action[:145])}</text>',
            f'<text class="body" x="60" y="{y + 116}">EXPECTED · {escape(decision.expected_effect[:145])}</text>',
            f'<text class="body" x="60" y="{y + 138}">VERIFY · {escape(decision.verify[:145])}</text>',
        ])
    parts.append("</svg>")
    return "".join(parts)


def _training_decision(code: str, training: Mapping[str, Any], add: Any) -> None:
    catalog = {
        "non_finite_metric": ("critical", "Stop non-finite training", "Check finite inputs, scaling, loss implementation, and learning rate before continuing.", "Finite metrics at every step.", "Run with anomaly detection and fail fast on the first non-finite value."),
        "metric_divergence": ("critical", "Restore the best checkpoint", "Roll back to the best step, reduce the learning rate, and enable clipping before adding capacity.", "Validation loss stops diverging.", "Compare the restored checkpoint against one lower-learning-rate run."),
        "metric_plateau": ("medium", "Respond to the plateau", "Try a learning-rate reduction or early stopping before increasing model size.", "Validation quality improves or the run ends at its best step.", "Compare scheduler versus early-stop runs with the same seed."),
        "probable_overfit": ("high", "Reduce generalization gap", "Keep the best validation checkpoint and test regularization, augmentation, or more representative data.", "Validation improves without relying on later training loss.", "Compare train/validation curves and calibration on a held-out slice."),
        "loss_regression": ("high", "Bisect the regression", "Compare the changed data, optimizer, architecture, and seed against the last healthy artifact one variable at a time.", "The regression is isolated or removed.", "Run an A/B matrix with one changed variable per run."),
        "missing_metric": ("high", "Restore observability", "Emit the primary and validation metrics at every evaluation step before judging the model.", "The report contains a complete comparable history.", "Reject runs with missing metric points in CI."),
    }
    item = catalog.get(code)
    if not item:
        return
    priority, title, action, expected, verify = item
    add(DecisionStep(code, "training", priority, title, ("training diagnostic: " + code,), action, expected, verify))


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None
