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

    @property
    def errors(self) -> List[Diagnostic]:
        return [item for item in self.diagnostics if item.severity == "error"]

    @property
    def warnings(self) -> List[Diagnostic]:
        return [item for item in self.diagnostics if item.severity == "warning"]

    def to_dict(self) -> Dict[str, Any]:
        return {"valid": self.valid, "diagnostics": [
            {"code": item.code, "severity": item.severity, "message": item.message,
             "details": dict(item.details)} for item in self.diagnostics
        ]}

    def summary(self) -> Dict[str, int]:
        return {severity: sum(item.severity == severity for item in self.diagnostics)
                for severity in ("error", "warning", "info")}

    def raise_if_invalid(self) -> "DiagnosticReport":
        if not self.valid:
            raise ValueError("; ".join(item.message for item in self.errors))
        return self
