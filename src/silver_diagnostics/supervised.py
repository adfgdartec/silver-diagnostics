"""Extensible catalog and assessment engine for supervised learning systems."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Tuple


@dataclass(frozen=True)
class LearningPrinciple:
    key: str
    title: str
    applies_to: Tuple[str, ...]
    checks: Tuple[str, ...]


@dataclass(frozen=True)
class DatasetRequirement:
    key: str
    description: str
    required_for: Tuple[str, ...]
    severity: str = "warning"


@dataclass(frozen=True)
class AlgorithmFamily:
    key: str
    title: str
    architectures: Tuple[str, ...]
    requirements: Tuple[str, ...]
    principles: Tuple[str, ...]


@dataclass(frozen=True)
class SupervisedAssessment:
    family: str
    task: str
    principles: Tuple[LearningPrinciple, ...]
    requirements: Tuple[DatasetRequirement, ...]
    blockers: Tuple[str, ...]
    warnings: Tuple[str, ...]
    recommended_visualizations: Tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.blockers

    def to_dict(self) -> Dict[str, Any]:
        return {"family": self.family, "task": self.task, "valid": self.valid,
                "principles": [item.key for item in self.principles],
                "requirements": [item.key for item in self.requirements],
                "blockers": list(self.blockers), "warnings": list(self.warnings),
                "recommended_visualizations": list(self.recommended_visualizations)}


PRINCIPLES = (
    LearningPrinciple("bias_variance", "Bias-variance tradeoff", ("all",), ("compare_train_validation", "learning_curves")),
    LearningPrinciple("empirical_risk", "Empirical risk minimization", ("all",), ("define_loss", "track_validation_loss")),
    LearningPrinciple("regularization", "Regularization and complexity control", ("all",), ("tune_regularization", "monitor_overfit")),
    LearningPrinciple("data_leakage", "Leakage prevention", ("all",), ("split_before_fit", "audit_provenance")),
    LearningPrinciple("class_balance", "Class balance and cost sensitivity", ("classification",), ("inspect_support", "choose_class_weights")),
    LearningPrinciple("calibration", "Probability calibration", ("classification",), ("check_reliability", "calibrate_outputs")),
    LearningPrinciple("margin_maximization", "Margin maximization", ("svm",), ("scale_features", "inspect_margins")),
    LearningPrinciple("locality", "Local similarity and neighborhood structure", ("knn",), ("scale_features", "inspect_distance")),
    LearningPrinciple("conditional_independence", "Conditional independence assumption", ("naive_bayes",), ("inspect_feature_dependence",)),
    LearningPrinciple("translation_equivariance", "Translation equivariance and locality", ("cnn",), ("preserve_spatial_layout", "augment_invariances")),
    LearningPrinciple("temporal_causality", "Temporal order and causality", ("rnn", "transformer", "forecasting"), ("time_split", "prevent_future_leakage")),
    LearningPrinciple("attention_context", "Context selection through attention", ("transformer",), ("bound_sequence_length", "inspect_attention")),
    LearningPrinciple("message_passing", "Graph message passing", ("gnn",), ("preserve_edges", "split_by_graph")),
    LearningPrinciple("reproducibility", "Reproducible experimentation", ("all",), ("record_seed", "fingerprint_data", "save_config")),
)

REQUIREMENTS = (
    DatasetRequirement("labels", "Every training row has a target label.", ("all",), "blocker"),
    DatasetRequirement("finite_features", "Features are finite and representable.", ("all",), "blocker"),
    DatasetRequirement("train_validation_test", "Splits are defined before fitting transforms.", ("all",), "blocker"),
    DatasetRequirement("class_support", "Every evaluated class has useful support.", ("classification",), "warning"),
    DatasetRequirement("scaled_numeric", "Numeric features are scaled when distances or margins matter.", ("svm", "knn", "mlp", "transformer")),
    DatasetRequirement("spatial_layout", "Spatial dimensions are preserved for convolutional models.", ("cnn",), "blocker"),
    DatasetRequirement("sequence_order", "Sequence order and time boundaries are available.", ("rnn", "transformer", "forecasting"), "blocker"),
    DatasetRequirement("graph_edges", "Nodes and edges are available for message passing.", ("gnn",), "blocker"),
    DatasetRequirement("provenance", "Source, split, and preprocessing provenance are recorded.", ("all",)),
)

FAMILIES = (
    AlgorithmFamily("linear", "Linear and logistic models", ("linear",), ("labels", "finite_features", "train_validation_test"), ("bias_variance", "empirical_risk", "data_leakage", "reproducibility")),
    AlgorithmFamily("tree", "Decision trees and random forests", ("tree", "forest"), ("labels", "finite_features", "train_validation_test"), ("bias_variance", "empirical_risk", "data_leakage", "class_balance")),
    AlgorithmFamily("boosting", "Gradient boosting", ("boosting",), ("labels", "finite_features", "train_validation_test"), ("bias_variance", "regularization", "data_leakage", "class_balance")),
    AlgorithmFamily("svm", "Support vector machines", ("svm",), ("labels", "finite_features", "train_validation_test", "scaled_numeric"), ("margin_maximization", "regularization", "data_leakage")),
    AlgorithmFamily("knn", "Nearest-neighbor methods", ("knn",), ("labels", "finite_features", "train_validation_test", "scaled_numeric"), ("locality", "data_leakage")),
    AlgorithmFamily("naive_bayes", "Naive Bayes", ("naive_bayes",), ("labels", "finite_features", "train_validation_test"), ("conditional_independence", "empirical_risk")),
    AlgorithmFamily("mlp", "Multilayer perceptrons", ("mlp",), ("labels", "finite_features", "train_validation_test", "scaled_numeric"), ("bias_variance", "regularization", "data_leakage")),
    AlgorithmFamily("cnn", "Convolutional networks", ("cnn",), ("labels", "finite_features", "train_validation_test", "spatial_layout"), ("translation_equivariance", "regularization", "data_leakage")),
    AlgorithmFamily("rnn", "Recurrent and temporal networks", ("rnn",), ("labels", "finite_features", "train_validation_test", "sequence_order"), ("temporal_causality", "regularization", "data_leakage")),
    AlgorithmFamily("transformer", "Transformer and GPT-style networks", ("transformer", "gpt"), ("labels", "finite_features", "train_validation_test", "sequence_order", "scaled_numeric"), ("attention_context", "temporal_causality", "regularization", "data_leakage", "calibration")),
    AlgorithmFamily("gnn", "Graph neural networks", ("gnn",), ("labels", "train_validation_test", "graph_edges"), ("message_passing", "data_leakage")),
)


def assess_supervised_system(family: str, task: str, dataset: Mapping[str, Any]) -> SupervisedAssessment:
    """Assess model-family requirements against a dataset capability profile."""
    selected = next((item for item in FAMILIES if item.key == family), None)
    if selected is None:
        raise ValueError("unknown algorithm family %r" % family)
    available = {str(key) for key, value in dataset.items() if value is True}
    principles = tuple(item for item in PRINCIPLES if item.key in selected.principles or "all" in item.applies_to or task in item.applies_to)
    requirements = tuple(item for item in REQUIREMENTS if item.key in selected.requirements or task in item.required_for or "all" in item.required_for)
    blockers = tuple("missing dataset capability: %s" % item.key for item in requirements if item.key in selected.requirements and item.severity == "blocker" and item.key not in available)
    warnings = tuple("missing or unverified dataset capability: %s" % item.key for item in requirements if item.key not in available and item.severity != "blocker")
    visuals = ["dataset_profile", "metric_history"]
    if task == "classification":
        visuals.extend(["class_balance", "confusion_matrix"])
    if family == "transformer":
        visuals.extend(["pipeline_graph", "runtime_trace"])
    return SupervisedAssessment(family, task, principles, requirements, blockers, warnings, tuple(dict.fromkeys(visuals)))
