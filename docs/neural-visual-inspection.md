# Diagnose the network, visually

Training health and layer health answer different questions. Training health
uses metric history to show convergence, divergence, plateau, overfitting, and
gradient trends. Layer health uses measured activation and gradient statistics
to flag dead or collapsed activations and vanishing or exploding gradients.

```python
from silver_diagnostics import diagnose_layer_health, diagnose_training_health

training = diagnose_training_health(history)
open("training-health.svg", "w", encoding="utf-8").write(training.to_svg(history))

layers = diagnose_layer_health(model_inspection.to_dict()["layers"])
open("layer-health.svg", "w", encoding="utf-8").write(layers.to_svg())
```

Every finding is threshold-based and machine-readable. Missing signals remain
unknown rather than being inferred. Layer reports use schema
`silver.diagnostics/layer-health-1`.

## From evidence to a decision

```python
from silver_diagnostics import build_debug_plan

plan = build_debug_plan(
    training=training.to_dict(),
    layers=model_inspection.to_dict()["layers"],
    neural_inputs=input_profile.to_dict(),
    topology=network.to_dict(),
)
open("decision-guide.svg", "w", encoding="utf-8").write(plan.to_svg())
```

Each decision includes observed evidence, a bounded intervention, the expected
effect, and a verification test. The planner proposes changes; it never silently
mutates the model, dataset, or optimizer.
