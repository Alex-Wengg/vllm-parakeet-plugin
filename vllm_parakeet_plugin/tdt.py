# SPDX-License-Identifier: Apache-2.0
"""Batched greedy TDT decoding over per-request slots.

Every request owns a slot: its projected encoder frames, a frame pointer and
the prediction network's LSTM state. One call to `decode_next_tokens`
advances any subset of slots to their next transcript token, consuming blanks
inside the loop, so the engine sees exactly one token per step.
"""

from dataclasses import dataclass

import torch
from torch import nn


class TDTSlots:
    """Greedy TDT decoding state for every request slot."""

    def __init__(
        self,
        *,
        num_slots: int,
        max_frames: int,
        hidden_size: int,
        num_layers: int,
        frames_dtype: torch.dtype,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.num_slots = num_slots
        self.max_frames = max_frames
        # One extra slot that never has frames, so it is never active. CUDA
        # graphs pad their batch with it.
        self.pad_slot = num_slots
        num_rows = num_slots + 1
        # Frames keep the encoder's dtype; the small recurrent state uses
        # `dtype` (the prediction and joint networks run in float32).
        self.frames = torch.zeros(
            num_rows, max_frames, hidden_size, dtype=frames_dtype, device=device
        )
        self.num_frames = torch.zeros(num_rows, dtype=torch.long, device=device)
        self.frame_idx = torch.zeros_like(self.num_frames)
        self.last_token = torch.zeros_like(self.num_frames)
        # Consecutive zero-duration emissions at the current frame.
        self.symbols = torch.zeros_like(self.num_frames)
        # Tokens to regenerate without emitting them (after a preemption).
        self.replay = torch.zeros_like(self.num_frames)
        # Whether the prediction network has consumed the blank seed yet.
        self.started = torch.zeros(num_rows, dtype=torch.bool, device=device)
        self.hidden = torch.zeros(
            num_layers, num_rows, hidden_size, dtype=dtype, device=device
        )
        self.cell = torch.zeros_like(self.hidden)
        self.decoder_out = torch.zeros(
            num_rows, hidden_size, dtype=dtype, device=device
        )

    def reset(
        self,
        slot: int,
        frames: torch.Tensor,
        *,
        blank_token_id: int,
        num_tokens_to_replay: int = 0,
    ) -> None:
        num_frames = frames.shape[0]
        if num_frames > self.max_frames:
            raise ValueError(
                f"Audio has {num_frames} encoder frames; at most "
                f"{self.max_frames} are supported."
            )
        self.frames[slot, :num_frames].copy_(frames)
        self.num_frames[slot] = num_frames
        self.frame_idx[slot] = 0
        self.last_token[slot] = blank_token_id
        self.symbols[slot] = 0
        self.replay[slot] = num_tokens_to_replay
        self.started[slot] = False
        self.hidden[:, slot] = 0
        self.cell[:, slot] = 0
        self.decoder_out[slot] = 0


@dataclass(frozen=True)
class TDTDecodeConfig:
    durations: torch.Tensor
    blank_token_id: int
    eos_token_id: int
    vocab_size: int
    max_symbols_per_step: int


def _active(
    state: TDTSlots, slots: torch.Tensor, pending: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slots still looking for a token, and slots that ran out of frames
    while replaying."""
    active = pending & (state.frame_idx[slots] < state.num_frames[slots])
    stuck = pending & ~active & (state.replay[slots] > 0)
    return active, stuck


def _advance(
    state: TDTSlots,
    slots: torch.Tensor,
    active: torch.Tensor,
    pending: torch.Tensor,
    tokens_out: torch.Tensor,
    decoder: nn.Module,
    joint: nn.Module,
    config: TDTDecodeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step of the loop for the `active` slots; the rest are unchanged.

    Returns the updated `pending` mask and output tokens.
    """
    frame_idx = state.frame_idx[slots]

    # Prediction network: one LSTM step on the previous token. Transformers
    # feeds the blank seed through it once, then keeps the state across
    # blanks, so only the first step and non-blank inputs update it.
    prev_tokens = state.last_token[slots]
    hidden = state.hidden[:, slots]
    cell = state.cell[:, slots]
    lstm_out, (new_hidden, new_cell) = decoder.lstm(
        decoder.embedding(prev_tokens).unsqueeze(1), (hidden, cell)
    )
    update = active & (~state.started[slots] | (prev_tokens != config.blank_token_id))
    state.hidden[:, slots] = torch.where(update[None, :, None], new_hidden, hidden)
    state.cell[:, slots] = torch.where(update[None, :, None], new_cell, cell)
    decoder_out = torch.where(
        update[:, None],
        decoder.decoder_projector(lstm_out[:, 0]),
        state.decoder_out[slots],
    )
    state.decoder_out[slots] = decoder_out
    state.started[slots] = state.started[slots] | active

    frames = state.frames[slots, frame_idx.clamp(max=state.max_frames - 1)]
    logits = joint(decoder_out, frames.to(decoder_out.dtype))
    tokens = logits[:, : config.vocab_size].argmax(dim=-1)
    steps = config.durations[logits[:, config.vocab_size :].argmax(dim=-1)]
    is_blank = tokens == config.blank_token_id
    steps = torch.where(is_blank & (steps == 0), 1, steps)
    symbols = torch.where(steps == 0, state.symbols[slots] + 1, 0)
    forced = symbols >= config.max_symbols_per_step
    steps = torch.where(forced, 1, steps)
    symbols = torch.where(forced, 0, symbols)

    emitted = active & ~is_blank
    replaying = emitted & (state.replay[slots] > 0)
    final = emitted & ~replaying
    state.frame_idx[slots] = torch.where(active, frame_idx + steps, frame_idx)
    state.symbols[slots] = torch.where(active, symbols, state.symbols[slots])
    state.last_token[slots] = torch.where(active, tokens, prev_tokens)
    state.replay[slots] = state.replay[slots] - replaying.long()
    return pending & ~final, torch.where(final, tokens, tokens_out)


@torch.inference_mode()
def decode_next_tokens(
    state: TDTSlots,
    slots: torch.Tensor,
    decoder: nn.Module,
    joint: nn.Module,
    *,
    durations: torch.Tensor,
    blank_token_id: int,
    eos_token_id: int,
    vocab_size: int,
    max_symbols_per_step: int,
) -> torch.Tensor:
    """Advance each slot to its next non-blank token.

    Follows Transformers' `ParakeetTDTGenerationMixin`: the token and the
    duration are separate argmaxes over the joint output, a blank predicted
    with duration 0 advances one frame, and the prediction network keeps its
    state across blanks. Like NeMo, a slot that emits `max_symbols_per_step`
    zero-duration tokens in a row is forced one frame ahead.

    Returns one token per slot: the next transcript token, or `eos_token_id`
    once the slot has walked past its last frame.
    """
    config = TDTDecodeConfig(
        durations=durations,
        blank_token_id=blank_token_id,
        eos_token_id=eos_token_id,
        vocab_size=vocab_size,
        max_symbols_per_step=max_symbols_per_step,
    )
    tokens_out = torch.full_like(slots, eos_token_id)
    pending = torch.ones_like(slots, dtype=torch.bool)
    while True:
        active, stuck = _active(state, slots, pending)
        any_active, any_stuck = torch.stack((active.any(), stuck.any())).tolist()
        if any_stuck:
            raise RuntimeError("TDT decoding ran out of frames while replaying.")
        if not any_active:
            return tokens_out
        pending, tokens_out = _advance(
            state, slots, active, pending, tokens_out, decoder, joint, config
        )


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    slots: torch.Tensor
    pending: torch.Tensor
    tokens_out: torch.Tensor
    stuck: torch.Tensor
    flags: torch.Tensor


class TDTGraphs:
    """`decode_next_tokens` as CUDA graphs, one per padded batch size.

    Each graph runs `iterations_per_graph` iterations of the loop. Extra
    iterations leave finished slots unchanged, so the host checks for
    unfinished slots once per replay instead of once per iteration, and the
    kernels of each replay launch together.
    """

    def __init__(
        self,
        state: TDTSlots,
        decoder: nn.Module,
        joint: nn.Module,
        config: TDTDecodeConfig,
        iterations_per_graph: int = 4,
    ) -> None:
        self.state = state
        self.decoder = decoder
        self.joint = joint
        self.config = config
        self.iterations_per_graph = iterations_per_graph
        sizes = [1]
        while sizes[-1] < state.num_slots:
            sizes.append(min(2 * sizes[-1], state.num_slots))
        self.batch_sizes = sizes
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: dict[int, _GraphEntry] = {}

    @torch.inference_mode()
    def decode(self, slots: torch.Tensor) -> torch.Tensor:
        num_slots = slots.numel()
        size = next(s for s in self.batch_sizes if s >= num_slots)
        entry = self.graphs.get(size)
        if entry is None:
            entry = self.graphs[size] = self._capture(size)
        entry.slots.fill_(self.state.pad_slot)
        entry.slots[:num_slots].copy_(slots)
        entry.pending.fill_(True)
        entry.tokens_out.fill_(self.config.eos_token_id)
        entry.stuck.fill_(False)
        while True:
            entry.graph.replay()
            any_active, any_stuck = entry.flags.tolist()
            if any_stuck:
                raise RuntimeError("TDT decoding ran out of frames while replaying.")
            if not any_active:
                return entry.tokens_out[:num_slots].clone()

    def _capture(self, size: int) -> _GraphEntry:
        device = self.state.num_frames.device
        # Every padded entry is the pad slot, which is never active, so the
        # warm-up run below leaves the live slots untouched.
        slots = torch.full(
            (size,), self.state.pad_slot, dtype=torch.long, device=device
        )
        pending = torch.ones(size, dtype=torch.bool, device=device)
        tokens_out = torch.full_like(slots, self.config.eos_token_id)
        stuck = torch.zeros_like(pending)
        flags = torch.zeros(2, dtype=torch.bool, device=device)

        def run() -> None:
            for _ in range(self.iterations_per_graph):
                active, now_stuck = _active(self.state, slots, pending)
                stuck.logical_or_(now_stuck)
                new_pending, new_tokens = _advance(
                    self.state,
                    slots,
                    active,
                    pending,
                    tokens_out,
                    self.decoder,
                    self.joint,
                    self.config,
                )
                pending.copy_(new_pending)
                tokens_out.copy_(new_tokens)
            active, _ = _active(self.state, slots, pending)
            flags.copy_(torch.stack((active.any(), stuck.any())))

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
            run()
        return _GraphEntry(graph, slots, pending, tokens_out, stuck, flags)
