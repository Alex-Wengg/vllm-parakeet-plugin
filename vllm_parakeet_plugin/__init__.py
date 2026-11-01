# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Parakeet TDT model plugin for vLLM."""


def register() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from .config import ParakeetForTDTConfig

    MODELS_CONFIG_MAP["ParakeetForTDT"] = ParakeetForTDTConfig
    ModelRegistry.register_model(
        "ParakeetForTDT", "vllm_parakeet_plugin.model:ParakeetForTDT"
    )
