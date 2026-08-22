"""Diagnostics for model blueprints and runtime resource requests."""

from typing import Any, Mapping, Optional

from .models import Diagnostic, DiagnosticReport


def diagnose_model_spec(spec: Mapping[str, Any]) -> DiagnosticReport:
    diagnostics = []
    for required in ("name", "architecture", "task"):
        if not spec.get(required):
            diagnostics.append(Diagnostic("missing_model_field", "error",
                "Model specification is missing %s." % required, {"field": required}))
    config = spec.get("config", {})
    if not isinstance(config, Mapping):
        diagnostics.append(Diagnostic("invalid_model_config", "error",
            "Model config must be a mapping.", {}))
    if spec.get("architecture") == "transformer" and isinstance(config, Mapping):
        vocab_size = _positive_integer(config.get("vocab_size"))
        hidden = _positive_integer(config.get("hidden_size"))
        heads = _positive_integer(config.get("heads"))
        if vocab_size is None:
            diagnostics.append(Diagnostic("invalid_vocab_size", "error",
                "Transformer vocab_size must be positive.", {"value": config.get("vocab_size")}))
        if config.get("hidden_size") is not None and hidden is None:
            diagnostics.append(Diagnostic("invalid_hidden_size", "error",
                "Transformer hidden_size must be a positive integer.",
                {"value": config.get("hidden_size")}))
        if config.get("heads") is not None and heads is None:
            diagnostics.append(Diagnostic("invalid_attention_heads", "error",
                "Transformer heads must be a positive integer.",
                {"value": config.get("heads")}))
        if hidden and heads and hidden % heads:
            diagnostics.append(Diagnostic("invalid_attention_shape", "error",
                "hidden_size must be divisible by heads.", {"hidden_size": hidden, "heads": heads}))
    return DiagnosticReport(not any(item.severity == "error" for item in diagnostics), diagnostics)


def _positive_integer(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value
