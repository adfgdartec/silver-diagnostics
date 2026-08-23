from .dataset import diagnose_dataset
from .classification import (
    ConfusionMatrixReport,
    HierarchicalConfusionReport,
    analyze_confusion_matrix,
    analyze_hierarchical_confusion,
)
from .metrics import diagnose_metric_history, diagnose_metrics
from .models import Diagnostic, DiagnosticReport
from .model import diagnose_model_spec as diagnose_model_spec
from .supervised import assess_supervised_system, AlgorithmFamily, DatasetRequirement, LearningPrinciple, SupervisedAssessment
from .training import Recommendation, TrainingHealthReport, diagnose_training_health
from .visualization import (
    LayerHealthReport,
    LayerHealthSignal,
    diagnose_layer_health,
    layer_health_svg,
    training_health_svg,
)
from .decision import DecisionPlan, DecisionStep, build_debug_plan, decision_plan_svg

__all__ = ["Diagnostic", "DiagnosticReport", "diagnose_dataset", "diagnose_metrics", "diagnose_metric_history",
           "ConfusionMatrixReport", "HierarchicalConfusionReport", "analyze_confusion_matrix",
           "analyze_hierarchical_confusion"]
__all__.append("diagnose_model_spec")
__all__ += ["LearningPrinciple", "DatasetRequirement", "AlgorithmFamily", "SupervisedAssessment", "assess_supervised_system"]
__all__ += ["Recommendation", "TrainingHealthReport", "diagnose_training_health"]
__all__ += [
    "LayerHealthReport", "LayerHealthSignal", "diagnose_layer_health",
    "layer_health_svg", "training_health_svg",
]
__all__ += ["DecisionPlan", "DecisionStep", "build_debug_plan", "decision_plan_svg"]

__version__ = "1.2.2"
