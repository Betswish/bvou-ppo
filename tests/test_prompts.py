from bvou_ppo.prompts import build_task_user_prompt, default_system_prompt
from bvou_ppo.registry import MODEL_REGISTRY


def test_boolq_prompt_builder():
    text, gold = build_task_user_prompt("boolq", {"passage": "abc", "question": "ok?", "answer": True})
    assert "Reply with only one word" in text
    assert gold == "yes"


def test_deepseek_system_prompt():
    prompt = default_system_prompt(MODEL_REGISTRY["deepseek_r1_0528_qwen3_8b"], "2026年3月19日，星期四")
    assert "DeepSeek-R1" in prompt
