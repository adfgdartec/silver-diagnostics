# Changelog

## 1.4.0 - 2026-08-23

- Added framework-neutral baseline/candidate report comparison with verdicts, metric deltas, and resolved/new decision codes.

## 1.3.0 - 2026-08-23

- Coordinated ecosystem release for validated, decision-ready experiment artifacts.

## 1.2.2 - 2026-08-23

- Keep the compatibility test focused on the public severity contract while exact-zero gradients remain visible in decision-plan evidence.

## 1.2.1 - 2026-08-23

- Preserve the established layer-health severity for exact-zero gradients while decision plans continue to surface them as vanishing-gradient evidence.

## 1.2.0 - 2026-08-23

- Added deterministic decision plans that rank data, architecture, optimization, and generalization fixes with verification steps.
- Added detailed decision-guide SVG output and zero-gradient vanishing-gradient detection.

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-08-22

### Added
- Added visual training-health reports with real loss curves, validation curves,
  scored findings, and per-layer gradient signals.
- Added `diagnose_layer_health()` for collapsed activations, dead activations,
  and vanishing or exploding gradients.

## [1.0.0] - 2026-08-22

### Added
- Added scored `TrainingHealthReport` analysis covering non-finite metrics,
  divergence, plateaus, probable overfitting, exploding gradients, and best-step
  selection.
- Added prioritized remediation actions, machine-readable reports, Markdown
  summaries, and critical-health policy enforcement.
- Added PEP 561 package markers and declared the stable public API.

## [0.4.0] - 2026-08-22

### Fixed
- Converted malformed transformer dimensions into structured diagnostics instead
  of leaking `TypeError` or `ValueError` exceptions.
- Distinguished invalid hidden sizes and attention-head counts in model reports.

### Added
- Hardened CI and releases with supported-Python testing and tag/version checks.

## [0.3.0] - 2026-08-15

### Changed
- Coordinated the package release with the supervised-learning architecture guide.

## [0.2.0] - 2026-08-15

### Changed
- Added report `errors`, `warnings`, `summary()`, `to_dict()`, and `raise_if_invalid()` helpers.
- Added required-metric thresholds and metric-history regression diagnostics.
- Added advanced confusion-matrix analysis with normalized matrices, per-class
  metrics, top confusions, and hierarchical parent-group errors.
- Added model-spec validation for transformer dimensions and task configuration.
- Added an extensible supervised-learning registry covering major principles,
  algorithm families, architectures, dataset requirements, blockers, warnings,
  and visualization recommendations.

## [0.1.0] - 2024-08-04

### Added
- Initial release of silver-diagnostics
- Framework-neutral dataset diagnostics
- Training metrics validation and analysis
- Non-finite value detection (NaN, infinity)
- Exploding gradient detection and warnings
- Dataset structure validation (empty checks, length mismatches, width consistency)
- Comprehensive test suite covering edge cases
- Support for Python 3.8-3.12

### Features
- `diagnose_dataset()` - Validate datasets for common issues
- `diagnose_metrics()` - Check training metrics for problems
- `Diagnostic` dataclass - Structured diagnostic information
- `DiagnosticReport` - Collection of diagnostics with validity status
- Error severity levels (error, warning, info)
- Detailed diagnostic information with context
- Specialized gradient analysis for training stability

## [Unreleased]
