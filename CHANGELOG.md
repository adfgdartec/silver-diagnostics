# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
