# SPDX-License-Identifier: Apache-2.0
"""Check the batched TDT loop against Transformers `generate` (CPU, fp32)."""

import io
import os

import numpy as np
import pytest
import torch

pytest.importorskip("soundfile")
import soundfile as sf
from transformers import AutoProcessor, ParakeetForTDT

from vllm_parakeet_plugin.tdt import TDTSlots, decode_next_tokens

MODEL = os.environ.get("PARAKEET_TDT_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
NUM_CLIPS = int(os.environ.get("PARAKEET_TDT_NUM_CLIPS", "8"))


def _librispeech_dummy(num_clips: int) -> list[np.ndarray]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean/validation-00000-of-00001.parquet",
        repo_type="dataset",
    )
    rows = pq.read_table(path).column("audio").to_pylist()[:num_clips]
    clips = []
    for row in rows:
        audio, sample_rate = sf.read(io.BytesIO(row["bytes"]), dtype="float32")
        assert sample_rate == 16_000
        clips.append(audio)
    return clips


@pytest.fixture(scope="module")
def model_and_processor():
    processor = AutoProcessor.from_pretrained(MODEL)
    model = ParakeetForTDT.from_pretrained(MODEL, dtype=torch.float32).eval()
    return model, processor


@pytest.fixture(scope="module")
def clips():
    return _librispeech_dummy(NUM_CLIPS)


def _reference_tokens(model, processor, audio: np.ndarray) -> list[int]:
    inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
    with torch.inference_mode():
        sequences = model.generate(**inputs).sequences[0].tolist()
    # Drop the decoder start token and every blank.
    blank = model.config.blank_token_id
    return [t for t in sequences[1:] if t != blank]


def _encoder_frames(model, processor, audio: np.ndarray) -> torch.Tensor:
    inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
    with torch.inference_mode():
        out = model.get_audio_features(
            input_features=inputs["input_features"],
            attention_mask=inputs["attention_mask"],
            output_attention_mask=True,
        )
    num_frames = int(out.attention_mask.sum())
    return out.pooler_output[0, :num_frames]


def _new_state(model, num_slots: int, max_frames: int) -> TDTSlots:
    config = model.config
    return TDTSlots(
        num_slots=num_slots,
        max_frames=max_frames,
        hidden_size=config.decoder_hidden_size,
        num_layers=config.num_decoder_layers,
        frames_dtype=torch.float32,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _step(model, state: TDTSlots, slots: torch.Tensor) -> list[int]:
    config = model.config
    return decode_next_tokens(
        state,
        slots,
        model.decoder,
        model.joint,
        durations=torch.tensor(config.durations),
        blank_token_id=config.blank_token_id,
        eos_token_id=model.generation_config.eos_token_id,
        vocab_size=config.vocab_size,
        max_symbols_per_step=config.max_symbols_per_step,
    ).tolist()


def _run_to_eos(model, state: TDTSlots, slots: list[int]) -> dict[int, list[int]]:
    eos = model.generation_config.eos_token_id
    outputs: dict[int, list[int]] = {slot: [] for slot in slots}
    active = list(slots)
    while active:
        tokens = _step(model, state, torch.tensor(active))
        still_active = []
        for slot, token in zip(active, tokens):
            if token == eos:
                continue
            outputs[slot].append(token)
            still_active.append(slot)
        active = still_active
    return outputs


def test_batched_loop_matches_transformers(model_and_processor, clips):
    model, processor = model_and_processor
    frames = [_encoder_frames(model, processor, audio) for audio in clips]
    state = _new_state(model, len(clips), max(f.shape[0] for f in frames))
    for slot, slot_frames in enumerate(frames):
        state.reset(slot, slot_frames, blank_token_id=model.config.blank_token_id)

    # All clips advance together, with different lengths in one batch.
    outputs = _run_to_eos(model, state, list(range(len(clips))))
    for slot, audio in enumerate(clips):
        expected = _reference_tokens(model, processor, audio)
        assert expected, "reference transcript is empty"
        assert outputs[slot] == expected, processor.decode(outputs[slot])


def test_replay_regenerates_prefix_silently(model_and_processor, clips):
    model, processor = model_and_processor
    audio = clips[0]
    frames = _encoder_frames(model, processor, audio)
    expected = _reference_tokens(model, processor, audio)
    num_replayed = len(expected) // 2

    state = _new_state(model, 1, frames.shape[0])
    state.reset(
        0,
        frames,
        blank_token_id=model.config.blank_token_id,
        num_tokens_to_replay=num_replayed,
    )
    assert _run_to_eos(model, state, [0])[0] == expected[num_replayed:]
