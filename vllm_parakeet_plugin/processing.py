# SPDX-License-Identifier: Apache-2.0
"""Audio preprocessing for Parakeet TDT."""

import math
from collections.abc import Mapping, Sequence

import torch
from transformers import BatchFeature, ParakeetFeatureExtractor, ParakeetTDTConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.renderers import TokenizeParams

# Longest audio a single request may carry. The transcription endpoint splits
# longer files into chunks of this size; pass
# `--hf-overrides '{"max_audio_clip_s": N}'` to change it.
DEFAULT_MAX_AUDIO_CLIP_S = 120


def get_max_audio_clip_s(hf_config: ParakeetTDTConfig) -> int:
    return int(getattr(hf_config, "max_audio_clip_s", DEFAULT_MAX_AUDIO_CLIP_S))


def subsampled_length(num_frames: int, encoder_config) -> int:
    """Mirror of `ParakeetEncoder._get_subsampling_output_length`."""
    kernel_size = encoder_config.subsampling_conv_kernel_size
    stride = encoder_config.subsampling_conv_stride
    add_pad = (kernel_size - 1) // 2 * 2 - kernel_size
    length = num_frames
    for _ in range(int(math.log2(encoder_config.subsampling_factor))):
        length = math.floor((length + add_pad) / stride + 1.0)
    return length


class ParakeetTDTProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> ParakeetTDTConfig:
        return self.ctx.get_hf_config(ParakeetTDTConfig)

    def get_default_tok_params(self) -> TokenizeParams:
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_feature_extractor(self, **kwargs: object) -> ParakeetFeatureExtractor:
        feature_extractor = self.get_hf_processor(**kwargs).feature_extractor
        assert isinstance(feature_extractor, ParakeetFeatureExtractor)
        return feature_extractor

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=self.get_feature_extractor().sampling_rate,
            target_channels=1,
        )

    @property
    def skip_prompt_length_check(self) -> bool:
        return True  # The encoder prompt holds one placeholder per frame.

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_max_audio_samples(self) -> int:
        sampling_rate = self.get_feature_extractor().sampling_rate
        return get_max_audio_clip_s(self.get_hf_config()) * sampling_rate

    def get_num_audio_tokens(self, num_samples: int) -> int:
        """Encoder frames for `num_samples` of audio."""
        feature_extractor = self.get_feature_extractor()
        num_mel_frames = num_samples // feature_extractor.hop_length
        return subsampled_length(num_mel_frames, self.get_hf_config().encoder_config)


class ParakeetTDTDummyInputsBuilder(BaseDummyInputsBuilder[ParakeetTDTProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        return {
            "audio": self._get_dummy_audios(
                length=self.info.get_max_audio_samples(),
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio"),
            )
        }


class ParakeetTDTMultiModalProcessor(
    EncDecMultiModalProcessor[ParakeetTDTProcessingInfo]
):
    # The decoder prompt is the blank seed, built below.
    skip_decoder_start_token: bool = True

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> list[int]:
        # Expanded to one placeholder per encoder frame.
        return [0]

    def create_decoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> list[int]:
        return [self.info.get_hf_config().blank_token_id]

    def _call_hf_processor(
        self,
        hf_data: Mapping[str, object],
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        feature_extractor = self.info.get_feature_extractor(**hf_kwargs)
        max_samples = self.info.get_max_audio_samples()
        audios = hf_data.get("audio")
        if audios is None:
            return BatchFeature()
        if not isinstance(audios, list):
            audios = [audios]

        # Extract one clip at a time so features are never padded to the
        # longest clip in the batch.
        input_features = []
        feature_lengths = []
        for audio in audios:
            if not feature_extractor.hop_length <= len(audio) <= max_samples:
                raise ValueError(
                    "Parakeet TDT takes between "
                    f"{feature_extractor.hop_length} and {max_samples} audio "
                    f"samples at {feature_extractor.sampling_rate} Hz per request."
                )
            features = feature_extractor(
                audio,
                sampling_rate=feature_extractor.sampling_rate,
                return_tensors="pt",
            )
            input_features.append(features["input_features"][0])
            feature_lengths.append(features["attention_mask"][0].sum())
        return BatchFeature(
            dict(
                input_features=input_features,
                feature_lengths=torch.stack(feature_lengths),
            )
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_features=MultiModalFieldConfig.batched("audio"),
            feature_lengths=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        audios = mm_items.get_items("audio", AudioProcessorItems)

        def get_replacement(item_idx: int) -> list[int]:
            num_samples = audios.get_audio_length(item_idx)
            return [0] * self.info.get_num_audio_tokens(num_samples)

        return [
            PromptReplacement(modality="audio", target=[0], replacement=get_replacement)
        ]
