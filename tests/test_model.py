from silver_diagnostics import diagnose_model_spec


def test_model_diagnostics_catch_attention_shape_errors():
    report = diagnose_model_spec({"name": "bad", "architecture": "transformer", "task": "classification",
                                  "config": {"vocab_size": 10, "hidden_size": 10, "heads": 3}})
    assert not report.valid
    assert any(item.code == "invalid_attention_shape" for item in report.errors)
