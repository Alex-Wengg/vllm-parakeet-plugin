# SPDX-License-Identifier: Apache-2.0
"""Check the batched TDT loop against Transformers `generate` (CPU, fp32)."""

import copy
import io
import os

import numpy as np
import pytest
import torch

pytest.importorskip("soundfile")
import soundfile as sf
from transformers import AutoProcessor, ParakeetForTDT

from vllm_parakeet_plugin.encoder_graphs import EncoderGraphs
from vllm_parakeet_plugin.tdt import (
    TDTDecodeConfig,
    TDTGraphs,
    TDTSlots,
    decode_next_tokens,
)

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


def _new_state(model, num_slots: int, max_frames: int, device: str = "cpu") -> TDTSlots:
    config = model.config
    return TDTSlots(
        num_slots=num_slots,
        max_frames=max_frames,
        hidden_size=config.decoder_hidden_size,
        num_layers=config.num_decoder_layers,
        frames_dtype=torch.float32,
        dtype=torch.float32,
        device=torch.device(device),
    )


def _decode_config(model, device: str = "cpu") -> TDTDecodeConfig:
    config = model.config
    return TDTDecodeConfig(
        durations=torch.tensor(config.durations, device=device),
        blank_token_id=config.blank_token_id,
        eos_token_id=model.generation_config.eos_token_id,
        vocab_size=config.vocab_size,
        max_symbols_per_step=config.max_symbols_per_step,
    )


def _eager_step(model, state: TDTSlots, decoder=None, joint=None, device="cpu"):
    config = _decode_config(model, device)
    decoder = model.decoder if decoder is None else decoder
    joint = model.joint if joint is None else joint

    def step(slots: torch.Tensor) -> torch.Tensor:
        return decode_next_tokens(state, slots, decoder, joint, **vars(config))

    return step


def _run_to_eos(
    model, state: TDTSlots, slots: list[int], step=None, device: str = "cpu"
) -> dict[int, list[int]]:
    eos = model.generation_config.eos_token_id
    step = _eager_step(model, state) if step is None else step
    outputs: dict[int, list[int]] = {slot: [] for slot in slots}
    active = list(slots)
    while active:
        tokens = step(torch.tensor(active, device=device)).tolist()
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


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)


@pytest.mark.gpu
@requires_cuda
def test_tdt_graphs_match_eager_loop(model_and_processor, clips):
    model, processor = model_and_processor
    frames = [_encoder_frames(model, processor, audio) for audio in clips]
    max_frames = max(f.shape[0] for f in frames)
    decoder = copy.deepcopy(model.decoder).cuda()
    joint = copy.deepcopy(model.joint).cuda()

    def transcribe(use_graphs: bool) -> dict[int, list[int]]:
        state = _new_state(model, len(clips), max_frames, device="cuda")
        for slot, slot_frames in enumerate(frames):
            state.reset(
                slot, slot_frames.cuda(), blank_token_id=model.config.blank_token_id
            )
        if use_graphs:
            graphs = TDTGraphs(state, decoder, joint, _decode_config(model, "cuda"))
            step = graphs.decode
        else:
            step = _eager_step(model, state, decoder, joint, device="cuda")
        return _run_to_eos(
            model, state, list(range(len(clips))), step=step, device="cuda"
        )

    assert transcribe(use_graphs=True) == transcribe(use_graphs=False)


@pytest.mark.gpu
@requires_cuda
def test_encoder_graphs_match_eager(model_and_processor, clips):
    model, processor = model_and_processor
    encoder = copy.deepcopy(model.encoder).cuda()
    projector = copy.deepcopy(model.encoder_projector).cuda()
    graphs = EncoderGraphs(
        encoder,
        projector,
        num_mel_bins=model.config.encoder_config.num_mel_bins,
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    features = []
    for audio in clips[:3]:
        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
        features.append(inputs["input_features"][0].cuda())
    lengths = torch.tensor([f.shape[0] for f in features], device="cuda")

    # Eager mode at the graph's padded shape. Padding to another length can
    # change cuDNN's TF32 rounding, like batching clips together already does.
    batch_size, num_frames = graphs._bucket(len(features), int(lengths.max()))
    padded = torch.zeros(batch_size, num_frames, features[0].shape[1], device="cuda")
    for i, clip in enumerate(features):
        padded[i, : clip.shape[0]] = clip
    padded_lengths = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    padded_lengths[: len(features)] = lengths
    mask = torch.arange(num_frames, device="cuda")[None, :] < padded_lengths[:, None]
    with torch.inference_mode():
        expected = projector(
            encoder(input_features=padded, attention_mask=mask).last_hidden_state
        )
    actual = graphs.run(features, lengths)
    for i, n in enumerate(encoder._get_subsampling_output_length(lengths).tolist()):
        torch.testing.assert_close(actual[i, :n], expected[i, :n], atol=0, rtol=0)
