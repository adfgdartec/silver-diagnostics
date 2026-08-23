"""Visual diagnostics for training curves and neural-network layer signals."""

from dataclasses import asdict, dataclass
from html import escape
import math
from typing import Any, Dict, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class LayerHealthSignal:
    name: str
    activation_std: float | None
    zero_fraction: float | None
    gradient_rms: float | None
    status: str
    findings: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["findings"] = list(self.findings)
        return value


@dataclass(frozen=True)
class LayerHealthReport:
    layers: Tuple[LayerHealthSignal, ...]
    score: int
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "silver.diagnostics/layer-health-1",
            "score": self.score,
            "status": self.status,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def to_svg(self, *, title: str = "Neural layer health") -> str:
        return layer_health_svg(self, title=title)


def diagnose_layer_health(
    layers: Sequence[Mapping[str, Any]],
    *,
    dead_zero_fraction: float = 0.95,
    vanishing_gradient: float = 1e-8,
    exploding_gradient: float = 100.0,
) -> LayerHealthReport:
    """Diagnose portable activation and gradient statistics from any framework."""
    if not 0 < dead_zero_fraction <= 1:
        raise ValueError("dead_zero_fraction must be in (0, 1]")
    values = []
    for index, layer in enumerate(layers):
        findings = []
        activation_std = _optional_number(layer.get("activation_std"))
        zero_fraction = _optional_number(layer.get("activation_zero_fraction", layer.get("zero_fraction")))
        gradient_rms = _optional_number(layer.get("gradient_rms"))
        if zero_fraction is not None and zero_fraction >= dead_zero_fraction:
            findings.append("mostly zero activations")
        if activation_std is not None and activation_std <= 1e-10:
            findings.append("collapsed activation variance")
        if gradient_rms is not None and 0 <= gradient_rms < vanishing_gradient:
            findings.append("vanishing gradient")
        if gradient_rms is not None and gradient_rms > exploding_gradient:
            findings.append("exploding gradient")
        severe = any("gradient" in item for item in findings)
        status = "critical" if severe else "warning" if findings else "healthy"
        values.append(LayerHealthSignal(
            str(layer.get("name", "layer-%d" % index)), activation_std,
            zero_fraction, gradient_rms, status, tuple(findings),
        ))
    critical = sum(layer.status == "critical" for layer in values)
    warnings = sum(layer.status == "warning" for layer in values)
    score = max(0, 100 - critical * 30 - warnings * 12)
    status = "critical" if critical else "warning" if warnings else "healthy"
    return LayerHealthReport(tuple(values), score, status)


def training_health_svg(report: Any, history: Sequence[Mapping[str, float]]) -> str:
    """Render loss, validation, gradient, findings, and next actions as one SVG."""
    width, height = 1080, 520
    colors = {"healthy": "#4de0a3", "warning": "#ffbd5c", "critical": "#ff6b7d"}
    status_color = colors.get(str(report.status), "#9fb0c3")
    chart_x, chart_y, chart_w, chart_h = 50, 150, 650, 250
    loss = _series(history, "loss")
    validation = _series(history, "val_loss") or _series(history, "validation_loss")
    gradients = _series(history, "gradient_norm")
    all_values = loss + validation
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Training health" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Inter,ui-sans-serif,system-ui,sans-serif}"
        ".title{fill:#f5f7fa;font-size:26px;font-weight:700}.meta{fill:#9fb0c3;font-size:12px}"
        ".section{fill:#dfe9f5;font-size:13px;font-weight:700}.finding{fill:#b9c7d6;font-size:12px}"
        ".grid{stroke:#223242;stroke-width:1}.loss{fill:none;stroke:#43d8ff;stroke-width:3}"
        ".val{fill:none;stroke:#9b7dff;stroke-width:3}.grad{fill:#43d8ff}</style>",
        '<rect width="1080" height="520" rx="22" fill="#0a1017"/>',
        '<text class="title" x="34" y="45">Training health</text>',
        f'<rect x="34" y="67" width="170" height="46" rx="12" fill="#111b25" stroke="{status_color}"/>',
        f'<text x="50" y="96" fill="{status_color}" font-size="16" font-weight="700">{escape(str(report.status).upper())} · {int(report.score)}/100</text>',
        f'<text class="meta" x="230" y="86">best {escape(str(report.primary_metric))}</text>',
        f'<text x="230" y="107" fill="#79e0ff" font-size="15" font-weight="700">{_format(report.best_value)} at step {report.best_step if report.best_step is not None else "—"}</text>',
    ]
    for step in range(6):
        y = chart_y + step * chart_h / 5
        parts.append(f'<line class="grid" x1="{chart_x}" y1="{y}" x2="{chart_x + chart_w}" y2="{y}"/>')
    parts.append(f'<polyline class="loss" points="{_points(loss, all_values, chart_x, chart_y, chart_w, chart_h)}"/>')
    parts.append(f'<polyline class="val" points="{_points(validation, all_values, chart_x, chart_y, chart_w, chart_h)}"/>')
    parts.extend([
        f'<text class="meta" x="{chart_x}" y="{chart_y + chart_h + 25}">LOSS</text>',
        f'<text x="{chart_x + 55}" y="{chart_y + chart_h + 25}" fill="#43d8ff" font-size="12">training</text>',
        f'<text x="{chart_x + 125}" y="{chart_y + chart_h + 25}" fill="#9b7dff" font-size="12">validation</text>',
        '<text class="section" x="740" y="154">FINDINGS</text>',
    ])
    findings = list(report.diagnostics)[:5]
    if findings:
        for index, item in enumerate(findings):
            parts.append(f'<text class="finding" x="740" y="{182 + index * 30}">• {escape(item.code)}: {escape(item.message[:42])}</text>')
    else:
        parts.append('<text class="finding" x="740" y="182">No training-health findings.</text>')
    parts.append('<text class="section" x="740" y="350">GRADIENT NORM</text>')
    max_gradient = max(gradients, default=1.0) or 1.0
    for index, value in enumerate(gradients[-18:]):
        bar = min(80.0, 80.0 * value / max_gradient)
        parts.append(f'<rect class="grad" x="{740 + index * 16}" y="{445 - bar}" width="10" height="{bar}" rx="2"/>')
    parts.append("</svg>")
    return "".join(parts)


def layer_health_svg(report: LayerHealthReport, *, title: str = "Neural layer health") -> str:
    height = 130 + max(1, len(report.layers)) * 64
    colors = {"healthy": "#4de0a3", "warning": "#ffbd5c", "critical": "#ff6b7d"}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="{escape(title)}" viewBox="0 0 980 {height}">',
        "<style>text{font-family:Inter,ui-sans-serif,system-ui,sans-serif}"
        ".title{fill:#f5f7fa;font-size:25px;font-weight:700}.meta{fill:#9fb0c3;font-size:12px}"
        ".name{fill:#e8eef7;font-size:13px;font-weight:700}</style>",
        f'<rect width="980" height="{height}" rx="22" fill="#0a1017"/>',
        f'<text class="title" x="34" y="44">{escape(title)}</text>',
        f'<text class="meta" x="34" y="68">{escape(report.status)} · score {report.score}/100 · activation and gradient signals</text>',
    ]
    for index, layer in enumerate(report.layers):
        y = 92 + index * 64
        color = colors[layer.status]
        finding = ", ".join(layer.findings) if layer.findings else "signals within thresholds"
        parts.extend([
            f'<rect x="30" y="{y}" width="920" height="48" rx="11" fill="#111b25" stroke="{color}"/>',
            f'<circle cx="49" cy="{y + 24}" r="6" fill="{color}"/>',
            f'<text class="name" x="65" y="{y + 20}">{escape(layer.name)}</text>',
            f'<text class="meta" x="65" y="{y + 38}">{escape(finding)}</text>',
            f'<text class="meta" x="610" y="{y + 29}">σ {_format(layer.activation_std)} · zero {_percent(layer.zero_fraction)} · grad {_format(layer.gradient_rms)}</text>',
        ])
    parts.append("</svg>")
    return "".join(parts)


def _series(history: Sequence[Mapping[str, float]], name: str) -> list[float]:
    return [float(item[name]) for item in history if name in item and _finite(item[name])]


def _points(values: Sequence[float], scale_values: Sequence[float], x: float, y: float,
            width: float, height: float) -> str:
    if not values:
        return ""
    minimum = min(scale_values, default=min(values))
    maximum = max(scale_values, default=max(values))
    span = maximum - minimum or 1.0
    return " ".join("%.1f,%.1f" % (
        x + index * width / max(1, len(values) - 1),
        y + height - (value - minimum) * height / span,
    ) for index, value in enumerate(values))


def _optional_number(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _format(value: Any) -> str:
    return "—" if not _finite(value) else f"{float(value):.3g}"


def _percent(value: Any) -> str:
    return "—" if not _finite(value) else f"{100 * float(value):.1f}%"
