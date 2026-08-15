"""Diagnostics for model blueprints and runtime resource requests."""

from typing import Any, Mapping

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
    if spec.get("architecture") == "transformer":
        if int(config.get("vocab_size", 0)) < 1:
            diagnostics.append(Diagnostic("invalid_vocab_size", "error",
                "Transformer vocab_size must be positive.", {"value": config.get("vocab_size")}))
        hidden = int(config.get("hidden_size", 0))
        heads = int(config.get("heads", 0))
        if hidden and heads and hidden % heads:
            diagnostics.append(Diagnostic("invalid_attention_shape", "error",
                "hidden_size must be divisible by heads.", {"hidden_size": hidden, "heads": heads}))
    return DiagnosticReport(not any(item.severity == "error" for item in diagnostics), diagnostics)
