# SPDX-License-Identifier: Apache-2.0
"""Model-specific config checks for Parakeet TDT."""

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)


class ParakeetForTDTConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        hf_config = model_config.hf_config
        # Encoder outputs are cached already projected to the prediction
        # network's width.
        model_config.model_arch_config.hidden_size = hf_config.decoder_hidden_size
        # The checkpoint keeps these two in generation_config.json only.
        generation_config = model_config.try_get_generation_config()
        hf_config.decoder_start_token_id = hf_config.blank_token_id
        if getattr(hf_config, "eos_token_id", None) is None:
            hf_config.eos_token_id = generation_config["eos_token_id"]
        # Greedy TDT picks one token per step and has no alternatives to score.
        if model_config.max_logprobs != 0:
            logger.info(
                "Parakeet TDT does not return logprobs; setting max_logprobs=0."
            )
            model_config.max_logprobs = 0

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        if not vllm_config.use_v2_model_runner:
            raise ValueError(
                "Parakeet TDT requires Model Runner V2; unset VLLM_USE_V2_MODEL_RUNNER."
            )
        model_config = vllm_config.model_config
        if not model_config.enforce_eager:
            # vLLM cannot graph-capture the data-dependent decode loop, so the
            # model captures its own CUDA graphs for the encoder and for fixed
            # runs of the loop. --enforce-eager turns those off too.
            logger.info(
                "Parakeet TDT captures its own CUDA graphs; "
                "setting enforce_eager=True for vLLM."
            )
            model_config.enforce_eager = True
            model_config.hf_config.parakeet_cuda_graphs = True
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.tensor_parallel_size != 1
            or parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError("Parakeet TDT supports only TP=1 and PP=1.")
        if model_config.quantization is not None:
            raise ValueError("Parakeet TDT does not support vLLM quantization.")
        if vllm_config.speculative_config is not None:
            raise ValueError("Parakeet TDT does not support speculative decoding.")
