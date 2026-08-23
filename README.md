# silver-diagnostics

<p align="center"><img src="https://raw.githubusercontent.com/adfgdartec/silver-diagnostics/main/docs/assets/silver-hero.png" alt="Silver actionable ML diagnostics" width="100%"></p>

**Turn training telemetry into a health score and a ranked next-action list.**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/adfgdartec/silver-diagnostics/actions/workflows/ci.yml/badge.svg)](https://github.com/adfgdartec/silver-diagnostics/actions/workflows/ci.yml)
[![Code Style](https://img.shields.io/badge/code%20style-flake8-blue.svg)](https://flake8.pycqa.org/)

Framework-neutral ML data and training diagnostics for Silver. A Python package designed for ML researchers who need robust data validation and training stability checks across different frameworks.

## Visual neural health

<p align="center"><img src="https://raw.githubusercontent.com/adfgdartec/silver-diagnostics/main/docs/assets/neural-network-inspection.png" alt="Neural-network inputs, hidden layers, activations, gradients, prediction, and training health" width="100%"></p>

```python
from silver_diagnostics import diagnose_layer_health, diagnose_training_health

training = diagnose_training_health(history)
open("training-health.svg", "w", encoding="utf-8").write(training.to_svg(history))

layers = diagnose_layer_health(model_inspection.to_dict()["layers"])
open("layer-health.svg", "w", encoding="utf-8").write(layers.to_svg())
```

Real curves expose convergence and overfitting; measured activation sparsity,
variance, and gradient RMS expose unhealthy layers. See the
[signal definitions and limits](docs/neural-visual-inspection.md).

## Installation

```bash
pip install silver-diagnostics
```

## Quick Start

### Training health, not just warnings

```python
from silver_diagnostics import diagnose_training_health

health = diagnose_training_health([
    {"loss": 1.0, "val_loss": 1.1},
    {"loss": 0.6, "val_loss": 0.7},
    {"loss": 0.4, "val_loss": 0.9},
])

print(health.status, health.score, health.best_step)
for action in health.recommendations:
    print(action.priority, action.title, action.action)

open("training-health.md", "w").write(health.to_markdown())
```

The health engine detects missing/non-finite metrics, exploding gradients,
plateaus, true divergence, regression, and probable overfitting. Reports are
machine-readable, Markdown-ready, and framework-neutral.

```python
from silver_diagnostics import diagnose_dataset, diagnose_metrics

# Diagnose dataset issues
features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
labels = [0.0, 1.0, 0.0]

report = diagnose_dataset(features, labels)
if not report.valid:
    print("Dataset issues found:")
    for diagnostic in report.diagnostics:
        print(f"  [{diagnostic.severity}] {diagnostic.message}")

# Diagnose metrics issues
metrics = {"loss": 0.5, "accuracy": 0.9, "gradient_norm": 1e6}
report = diagnose_metrics(metrics)
for diagnostic in report.diagnostics:
    print(f"[{diagnostic.severity}] {diagnostic.message}")
```

### Deep classification analysis

```python
from silver_diagnostics import analyze_confusion_matrix

report = analyze_confusion_matrix(
    [[92, 6, 2], [8, 84, 8], [1, 9, 90]],
    labels=["healthy", "warning", "failure"],
    hierarchy={"healthy": "state", "warning": "state", "failure": "state"},
)
print(report.macro_f1, report.top_confusions)
```

Reports include per-class precision, recall, F1, specificity, support,
normalized matrices, top error pairs, and optional parent-group confusion.
Use `analyze_hierarchical_confusion()` for coarse-to-fine levels.

### Supervised-learning readiness

Use the registry to assess architecture-specific dataset requirements before
training:

```python
from silver_diagnostics import assess_supervised_system

assessment = assess_supervised_system("transformer", "classification", {
    "labels": True, "finite_features": True,
    "train_validation_test": True, "sequence_order": True,
    "scaled_numeric": True,
})
if not assessment.valid:
    raise ValueError(assessment.blockers)
```

## Features

- **Dataset Validation**: Comprehensive checks for empty data, length mismatches, and structural issues
- **Non-Finite Detection**: Automatic detection of NaN and infinity values in features and labels
- **Metrics Diagnostics**: Training metrics validation for numerical stability
- **Exploding Gradient Detection**: Specialized checks for gradient explosion during training
- **Framework-Agnostic**: Works with PyTorch, TensorFlow, JAX, or any numeric data
- **Detailed Reporting**: Structured diagnostic information with severity levels and context
- **Type Safety**: Full type hints for better IDE support and fewer bugs

## Use Cases

### Training Pipeline Validation

```python
from silver_diagnostics import diagnose_dataset, diagnose_metrics
import torch

# Validate training data before training
train_features = torch.randn(1000, 10).numpy()
train_labels = torch.randint(0, 2, (1000,)).numpy()

report = diagnose_dataset(train_features.tolist(), train_labels.tolist())
if not report.valid:
    print("Cannot train with invalid dataset:")
    for diagnostic in report.diagnostics:
        print(f"  {diagnostic.code}: {diagnostic.message}")
else:
    print("Dataset is valid for training")
```

### Training Stability Monitoring

```python
from silver_diagnostics import diagnose_metrics

# Monitor training metrics for stability
def check_training_stability(metrics):
    report = diagnose_metrics(metrics)
    
    # Check for errors
    errors = [d for d in report.diagnostics if d.severity == "error"]
    if errors:
        print("Training stability issues:")
        for error in errors:
            print(f"  {error.code}: {error.message}")
            return False
    
    # Check for warnings
    warnings = [d for d in report.diagnostics if d.severity == "warning"]
    if warnings:
        print("Training stability warnings:")
        for warning in warnings:
            print(f"  {warning.code}: {warning.message}")
    
    return True

# During training loop
for epoch in range(10):
    loss = train_epoch()
    metrics = {
        "loss": loss,
        "gradient_norm": compute_gradient_norm(),
        "accuracy": evaluate()
    }
    
    if not check_training_stability(metrics):
        print("Training unstable - stopping")
        break
```

### Data Quality Assurance

```python
from silver_diagnostics import diagnose_dataset

def validate_ml_pipeline_data(X_train, y_train, X_val, y_val):
    """Validate all datasets in ML pipeline"""
    datasets = {
        "training": (X_train, y_train),
        "validation": (X_val, y_val)
    }
    
    all_valid = True
    for name, (features, labels) in datasets.items():
        report = diagnose_dataset(features.tolist(), labels.tolist())
        
        print(f"\n{name} dataset:")
        if report.valid:
            print(f"  ✓ Valid ({len(features)} samples)")
        else:
            print(f"  ✗ Invalid")
            for diagnostic in report.diagnostics:
                print(f"    {diagnostic.message}")
            all_valid = False
    
    return all_valid
```

### Framework Integration

```python
from silver_diagnostics import diagnose_dataset, diagnose_metrics
import tensorflow as tf
import torch

# Works with TensorFlow tensors
tf_features = tf.random.normal((100, 10))
tf_labels = tf.random.uniform((100,), maxval=2, dtype=tf.int32)

report = diagnose_dataset(
    tf_features.numpy().tolist(),
    tf_labels.numpy().tolist()
)

# Works with PyTorch tensors
torch_features = torch.randn(100, 10)
torch_labels = torch.randint(0, 2, (100,))

report = diagnose_dataset(
    torch_features.tolist(),
    torch_labels.tolist()
)
```

## Advanced Usage

### Custom Diagnostic Processing

```python
from silver_diagnostics import diagnose_dataset, Diagnostic

def categorize_diagnostics(report):
    """Categorize diagnostics by type"""
    categories = {
        "structural": [],
        "data_quality": [],
        "numerical": []
    }
    
    for diagnostic in report.diagnostics:
        if diagnostic.code in ["empty_dataset", "length_mismatch", "feature_width_mismatch"]:
            categories["structural"].append(diagnostic)
        elif diagnostic.code in ["non_finite_feature", "non_finite_label"]:
            categories["numerical"].append(diagnostic)
        else:
            categories["data_quality"].append(diagnostic)
    
    return categories

report = diagnose_dataset(features, labels)
categories = categorize_diagnostics(report)

for category, diagnostics in categories.items():
    if diagnostics:
        print(f"{category.upper()} ({len(diagnostics)}):")
        for diag in diagnostics:
            print(f"  - {diag.message}")
```

### Batch Validation

```python
from silver_diagnostics import diagnose_dataset

def validate_multiple_datasets(dataset_dict):
    """Validate multiple datasets at once"""
    results = {}
    
    for name, (features, labels) in dataset_dict.items():
        report = diagnose_dataset(features, labels)
        results[name] = {
            "valid": report.valid,
            "error_count": sum(1 for d in report.diagnostics if d.severity == "error"),
            "warning_count": sum(1 for d in report.diagnostics if d.severity == "warning"),
            "diagnostics": report.diagnostics
        }
    
    return results

datasets = {
    "train": (X_train.tolist(), y_train.tolist()),
    "val": (X_val.tolist(), y_val.tolist()),
    "test": (X_test.tolist(), y_test.tolist())
}

validation_results = validate_multiple_datasets(datasets)
for name, result in validation_results.items():
    status = "✓" if result["valid"] else "✗"
    print(f"{status} {name}: {result['error_count']} errors, {result['warning_count']} warnings")
```

## Requirements

- Python 3.10+

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=silver_diagnostics --cov-report=html

# Run linting
flake8 src/ tests/
mypy src/
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache-2.0 - see [LICENSE](LICENSE) file for details.

## Related Packages

- [silver-data](https://github.com/adfgdartec/silver-data) - Dataset handling
- [silver-run](https://github.com/adfgdartec/silver-run) - Training lifecycle
- [silver-adapters](https://github.com/adfgdartec/silver-adapters) - Framework adapters

## From visuals to fixes

`build_debug_plan(...)` combines training health, layer signals, input
profiles, and topology into a deterministic decision plan. Each ranked step
contains its evidence, a concrete action, the expected effect, and a
verification test; `DecisionPlan.to_svg()` makes the full reasoning reviewable
in code review or experiment artifacts.
