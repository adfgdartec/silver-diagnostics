from silver_diagnostics import (
    diagnose_layer_health,
    diagnose_training_health,
    training_health_svg,
)


def test_training_health_svg_contains_real_curves_findings_and_gradients():
    history = [
        {"loss": 1.0, "val_loss": 1.1, "gradient_norm": 2.0},
        {"loss": 0.6, "val_loss": 0.7, "gradient_norm": 1.0},
        {"loss": 0.3, "val_loss": 0.4, "gradient_norm": 0.5},
    ]
    report = diagnose_training_health(history)
    svg = training_health_svg(report, history)
    assert svg.startswith("<svg")
    assert "Training health" in svg
    assert "GRADIENT NORM" in svg
    assert 'class="loss"' in svg
    assert report.to_svg(history) == svg


def test_layer_health_diagnoses_dead_and_exploding_layers_visually():
    report = diagnose_layer_health([
        {"name": "encoder", "activation_std": 0.4,
         "activation_zero_fraction": 0.2, "gradient_rms": 0.1},
        {"name": "dead-relu", "activation_std": 0.0,
         "activation_zero_fraction": 0.99, "gradient_rms": 0.0},
        {"name": "head", "activation_std": 2.0,
         "activation_zero_fraction": 0.0, "gradient_rms": 500.0},
    ])
    assert report.status == "critical"
    assert report.layers[1].status == "warning"
    assert "exploding gradient" in report.layers[2].findings
    assert report.to_dict()["schema"] == "silver.diagnostics/layer-health-1"
    svg = report.to_svg()
    assert "dead-relu" in svg
    assert "exploding gradient" in svg
