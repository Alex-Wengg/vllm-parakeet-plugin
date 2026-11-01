# SPDX-License-Identifier: Apache-2.0
"""Batched greedy TDT decoding over per-request slots.

Every request owns a slot: its projected encoder frames, a frame pointer and
the prediction network's LSTM state. One call to `decode_next_tokens`
advances any subset of slots to their next transcript token, consuming blanks
inside the loop, so the engine sees exactly one token per step.
"""

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
        self.max_frames = max_frames
        # Frames keep the encoder's dtype; the small recurrent state uses
        # `dtype` (the prediction and joint networks run in float32).
        self.frames = torch.zeros(
            num_slots, max_frames, hidden_size, dtype=frames_dtype, device=device
        )
        self.num_frames = torch.zeros(num_slots, dtype=torch.long, device=device)
        self.frame_idx = torch.zeros_like(self.num_frames)
        self.last_token = torch.zeros_like(self.num_frames)
        # Consecutive zero-duration emissions at the current frame.
        self.symbols = torch.zeros_like(self.num_frames)
        # Tokens to regenerate without emitting them (after a preemption).
        self.replay = torch.zeros_like(self.num_frames)
        # Whether the prediction network has consumed the blank seed yet.
        self.started = torch.zeros(num_slots, dtype=torch.bool, device=device)
        self.hidden = torch.zeros(
            num_layers, num_slots, hidden_size, dtype=dtype, device=device
        )
        self.cell = torch.zeros_like(self.hidden)
        self.decoder_out = torch.zeros(
            num_slots, hidden_size, dtype=dtype, device=device
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
    tokens_out = torch.full_like(slots, eos_token_id)
    pending = torch.ones_like(slots, dtype=torch.bool)
    max_frame = state.max_frames - 1
    while True:
        frame_idx = state.frame_idx[slots]
        active = pending & (frame_idx < state.num_frames[slots])
        stuck = pending & ~active & (state.replay[slots] > 0)
        any_active, any_stuck = torch.stack((active.any(), stuck.any())).tolist()
        if any_stuck:
            raise RuntimeError("TDT decoding ran out of frames while replaying.")
        if not any_active:
            return tokens_out

        # Prediction network: one LSTM step on the previous token. Transformers
        # feeds the blank seed through it once, then keeps the state across
        # blanks, so only the first step and non-blank inputs update it.
        prev_tokens = state.last_token[slots]
        hidden = state.hidden[:, slots]
        cell = state.cell[:, slots]
        lstm_out, (new_hidden, new_cell) = decoder.lstm(
            decoder.embedding(prev_tokens).unsqueeze(1), (hidden, cell)
        )
        update = active & (~state.started[slots] | (prev_tokens != blank_token_id))
        state.hidden[:, slots] = torch.where(update[None, :, None], new_hidden, hidden)
        state.cell[:, slots] = torch.where(update[None, :, None], new_cell, cell)
        decoder_out = torch.where(
            update[:, None],
            decoder.decoder_projector(lstm_out[:, 0]),
            state.decoder_out[slots],
        )
        state.decoder_out[slots] = decoder_out
        state.started[slots] = state.started[slots] | active

        frames = state.frames[slots, frame_idx.clamp(max=max_frame)]
        logits = joint(decoder_out, frames.to(decoder_out.dtype))
        tokens = logits[:, :vocab_size].argmax(dim=-1)
        steps = durations[logits[:, vocab_size:].argmax(dim=-1)]
        is_blank = tokens == blank_token_id
        steps = torch.where(is_blank & (steps == 0), 1, steps)
        symbols = torch.where(steps == 0, state.symbols[slots] + 1, 0)
        forced = symbols >= max_symbols_per_step
        steps = torch.where(forced, 1, steps)
        symbols = torch.where(forced, 0, symbols)

        emitted = active & ~is_blank
        replaying = emitted & (state.replay[slots] > 0)
        final = emitted & ~replaying
        state.frame_idx[slots] = torch.where(active, frame_idx + steps, frame_idx)
        state.symbols[slots] = torch.where(active, symbols, state.symbols[slots])
        state.last_token[slots] = torch.where(active, tokens, prev_tokens)
        state.replay[slots] = state.replay[slots] - replaying.long()
        tokens_out = torch.where(final, tokens, tokens_out)
        pending = pending & ~final
