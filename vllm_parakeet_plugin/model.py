# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Parakeet TDT transcription.

The FastConformer encoder, the LSTM prediction network and the joint network
are the Transformers modules, loaded from the native checkpoint. The greedy
TDT loop runs inside `forward` and hands the engine one finished token per
step; see `tdt.py`.
"""

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import nn
from transformers import ParakeetEncoder, ParakeetTDTConfig
from transformers.models.parakeet.modeling_parakeet import (
    ParakeetRNNTDecoder,
    ParakeetTDTJointNetwork,
)
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import (
    ExplicitEncoderDecoderPrompt,
    PromptType,
    TextPrompt,
    TokensPrompt,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsTranscription,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils.collection_utils import is_list_of

from .processing import (
    ParakeetTDTDummyInputsBuilder,
    ParakeetTDTMultiModalProcessor,
    ParakeetTDTProcessingInfo,
    get_max_audio_clip_s,
    subsampled_length,
)
from .tdt import TDTSlots, decode_next_tokens

if TYPE_CHECKING:
    from .model_state import ParakeetTDTModelState

# See https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3. The model detects
# the language itself; `language` is only validated.
PARAKEET_TDT_V3_LANGS = {
    "bg": "Bulgarian",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "it": "Italian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mt": "Maltese",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "ru": "Russian",
    "uk": "Ukrainian",
}


@MULTIMODAL_REGISTRY.register_processor(
    ParakeetTDTMultiModalProcessor,
    info=ParakeetTDTProcessingInfo,
    dummy_inputs=ParakeetTDTDummyInputsBuilder,
)
class ParakeetForTDT(nn.Module, SupportsTranscription, SupportsMultiModal):
    supported_languages: ClassVar[Mapping[str, str]] = PARAKEET_TDT_V3_LANGS
    supports_transcription_only = True

    # The convolution module's BatchNorm counter is not a parameter.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_suffix={".num_batches_tracked": None})

    @staticmethod
    def get_model_state_cls() -> type["ParakeetTDTModelState"]:
        from .model_state import ParakeetTDTModelState

        return ParakeetTDTModelState

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        return SpeechToTextConfig(
            sample_rate=16_000,
            max_audio_clip_s=get_max_audio_clip_s(model_config.hf_config),
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        # ParakeetFeatureExtractor uses a 160-sample hop.
        num_mel_frames = int(audio_duration_s * stt_config.sample_rate) // 160
        return subsampled_length(num_mel_frames, model_config.hf_config.encoder_config)

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        if stt_params.task_type != "transcribe" or stt_params.to_language:
            raise ValueError("Parakeet TDT supports transcription only.")
        if stt_params.request_prompt or stt_params.hotwords:
            raise ValueError("Parakeet TDT does not take a text prompt or hotwords.")
        return ExplicitEncoderDecoderPrompt(
            encoder_prompt=TextPrompt(
                prompt="",
                multi_modal_data={
                    "audio": (stt_params.audio, stt_params.stt_config.sample_rate)
                },
            ),
            decoder_prompt=TokensPrompt(
                prompt_token_ids=[stt_params.model_config.hf_config.blank_token_id]
            ),
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: ParakeetTDTConfig = vllm_config.model_config.hf_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype
        config.encoder_config._attn_implementation = "sdpa"
        self.encoder = ParakeetEncoder(config.encoder_config).to(self.dtype)
        self.encoder_projector = nn.Linear(
            config.encoder_config.hidden_size, config.decoder_hidden_size
        )
        # The prediction and joint networks are tiny and pick every token;
        # keep them in float32 whatever the encoder's dtype.
        self.decoder = ParakeetRNNTDecoder(config).float()
        self.joint = ParakeetTDTJointNetwork(config).float()

        self.eos_token_id = vllm_config.model_config.hf_config.eos_token_id
        self.register_buffer(
            "durations", torch.tensor(config.durations), persistent=False
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        input_features = kwargs.pop("input_features", None)
        feature_lengths = kwargs.pop("feature_lengths", None)
        if input_features is None:
            return []
        if not isinstance(feature_lengths, torch.Tensor):
            raise ValueError("Audio feature lengths must be a tensor.")
        if isinstance(input_features, torch.Tensor):
            features = list(input_features)
        elif is_list_of(input_features, torch.Tensor):
            features = input_features
        else:
            raise ValueError(
                f"Incorrect type of audio features: {type(input_features)}"
            )

        padded = nn.utils.rnn.pad_sequence(
            [f.to(self.dtype) for f in features], batch_first=True, padding_value=0.0
        )
        feature_lengths = feature_lengths.reshape(-1).to(padded.device)
        attention_mask = (
            torch.arange(padded.shape[1], device=padded.device)[None, :]
            < feature_lengths[:, None]
        )
        hidden_states = self.encoder(
            input_features=padded, attention_mask=attention_mask
        ).last_hidden_state
        frames = self.encoder_projector(hidden_states)
        num_frames = self.encoder._get_subsampling_output_length(feature_lengths)
        return [f[:n] for f, n in zip(frames, num_frames.tolist())]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        tdt_state: TDTSlots | None = None,
        tdt_slots: torch.Tensor | None = None,
        tdt_rows: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # One "hidden state" per token: the id of the token the engine should
        # sample there. Rows that are not finishing a prefill keep the eos
        # placeholder; the engine discards them.
        selected = torch.full(
            (input_ids.shape[0], 1),
            self.eos_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        if tdt_state is not None and tdt_slots is not None and tdt_slots.numel():
            assert tdt_rows is not None
            selected[tdt_rows, 0] = decode_next_tokens(
                tdt_state,
                tdt_slots,
                self.decoder,
                self.joint,
                durations=self.durations,
                blank_token_id=self.config.blank_token_id,
                eos_token_id=self.eos_token_id,
                vocab_size=self.config.vocab_size,
                max_symbols_per_step=self.config.max_symbols_per_step,
            )
        return selected

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The TDT loop already picked each token; make the sampler keep it.
        # These are not model probabilities.
        logits = torch.full(
            (hidden_states.shape[0], self.config.vocab_size),
            float("-inf"),
            dtype=self.dtype,
            device=hidden_states.device,
        )
        return logits.scatter_(1, hidden_states.long(), 0.0)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )
