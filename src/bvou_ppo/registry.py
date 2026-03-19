from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    hf_id: str
    family: str
    default_enable_thinking: bool
    supports_enable_thinking: bool
    use_official_system_prompt: bool
    recommended_temperature: float
    recommended_top_p: float


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen35_4b": ModelSpec(
        alias="qwen35_4b",
        hf_id="Qwen/Qwen3.5-4B",
        family="qwen35",
        default_enable_thinking=False,
        supports_enable_thinking=True,
        use_official_system_prompt=False,
        recommended_temperature=0.7,
        recommended_top_p=0.8,
    ),
    "qwen35_9b": ModelSpec(
        alias="qwen35_9b",
        hf_id="Qwen/Qwen3.5-9B",
        family="qwen35",
        default_enable_thinking=False,
        supports_enable_thinking=True,
        use_official_system_prompt=False,
        recommended_temperature=0.7,
        recommended_top_p=0.8,
    ),
    "deepseek_r1_0528_qwen3_8b": ModelSpec(
        alias="deepseek_r1_0528_qwen3_8b",
        hf_id="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        family="deepseek_r1_0528",
        default_enable_thinking=True,
        supports_enable_thinking=False,
        use_official_system_prompt=True,
        recommended_temperature=0.6,
        recommended_top_p=0.95,
    ),
}


def resolve_model(model_alias_or_id: str) -> ModelSpec:
    return MODEL_REGISTRY.get(
        model_alias_or_id,
        ModelSpec(
            alias=model_alias_or_id,
            hf_id=model_alias_or_id,
            family="generic",
            default_enable_thinking=False,
            supports_enable_thinking=False,
            use_official_system_prompt=False,
            recommended_temperature=0.7,
            recommended_top_p=0.9,
        ),
    )
