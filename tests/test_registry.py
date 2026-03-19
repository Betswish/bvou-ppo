from beippo.registry import resolve_model


def test_resolve_model_by_alias():
    spec = resolve_model("qwen35_4b")
    assert spec.family == "qwen35"
    assert spec.supports_enable_thinking is True


def test_resolve_model_by_hf_id():
    spec = resolve_model("Qwen/Qwen3.5-4B")
    assert spec.family == "qwen35"
    assert spec.supports_enable_thinking is True
