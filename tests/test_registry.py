from beippo.registry import resolve_model


def test_qwen_registry():
    spec = resolve_model("qwen35_4b")
    assert spec.hf_id == "Qwen/Qwen3.5-4B"
    assert spec.supports_enable_thinking is True
