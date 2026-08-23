from silver_diagnostics import build_debug_plan


def test_debug_plan_turns_cross_package_evidence_into_ranked_actions():
    plan = build_debug_plan(
        training={
            "status": "warning",
            "score": 72,
            "diagnostics": [{"code": "probable_overfit"}],
        },
        layers=[{
            "name": "encoder.relu",
            "activation_zero_fraction": 0.99,
            "activation_std": 0.0,
            "gradient_rms": 0.0,
        }],
        neural_inputs={"features": [{
            "name": "country", "role": "categorical", "unique_values": 80,
            "missing_rate": 0.3,
        }]},
    )
    assert plan.status == "warning"
    assert plan.decisions[0].priority == "high"
    codes = {item.code for item in plan.decisions}
    assert {"dead_activation", "vanishing_gradient", "repair_missingness", "high_cardinality"} <= codes
    assert "VERIFY" in plan.to_svg()
    assert plan.to_dict()["schema"] == "silver.diagnostics/decision-plan-1"


def test_debug_plan_preserves_a_healthy_baseline():
    plan = build_debug_plan(training={"status": "healthy", "score": 100})
    assert plan.status == "healthy"
    assert not plan.decisions
    assert "baseline" in plan.next_experiment


def test_training_report_can_become_a_decision_plan():
    from silver_diagnostics import diagnose_training_health

    report = diagnose_training_health([
        {"loss": 1.0, "val_loss": 1.1},
        {"loss": 0.8, "val_loss": 1.4},
        {"loss": 0.7, "val_loss": 1.6},
    ])
    plan = report.to_decision_plan()
    assert plan.decisions
    assert any(item.code == "probable_overfit" for item in plan.decisions)
