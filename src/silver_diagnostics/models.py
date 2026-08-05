from dataclasses import dataclass
from typing import Any, Dict, List, Literal


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: Literal["error", "warning", "info"]
    message: str
    details: Dict[str, Any]


@dataclass(frozen=True)
class DiagnosticReport:
    valid: bool
    diagnostics: List[Diagnostic]
