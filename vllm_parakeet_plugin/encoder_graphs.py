# SPDX-License-Identifier: Apache-2.0
"""The Parakeet encoder as CUDA graphs at padded input shapes."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    features: torch.Tensor
    lengths: torch.Tensor
    output: torch.Tensor


class EncoderGraphs:
    """Encoder and projector as CUDA graphs, captured per padded batch shape.

    A batch is padded to a power-of-two number of clips and a multiple of
    `frame_step` mel frames. The padding is masked like the padding eager mode
    adds to the longest clip, so the frames match up to kernel rounding.
    Batches over `max_batch_frames` padded frames run eagerly instead; at that
    size, launch overhead matters little.
    """

    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        *,
        num_mel_bins: int,
        dtype: torch.dtype,
        device: torch.device,
        max_batch_frames: int = 16384,
        frame_step: int = 512,
    ) -> None:
        self.encoder = encoder
        self.projector = projector
        self.num_mel_bins = num_mel_bins
        self.max_batch_frames = max_batch_frames
        self.frame_step = frame_step
        # Inputs for every graph are views into these buffers.
        self.features = torch.zeros(
            max_batch_frames, num_mel_bins, dtype=dtype, device=device
        )
        self.lengths = torch.zeros(
            max_batch_frames // frame_step, dtype=torch.long, device=device
        )
        self.positions = torch.arange(max_batch_frames, device=device)
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: dict[tuple[int, int], _GraphEntry] = {}

    def _bucket(self, batch_size: int, num_frames: int) -> tuple[int, int] | None:
        padded_batch = 1 << (batch_size - 1).bit_length()
        padded_frames = -(-num_frames // self.frame_step) * self.frame_step
        if padded_batch * padded_frames > self.max_batch_frames:
            return None
        return padded_batch, padded_frames

    @torch.inference_mode()
    def run(
        self, features: list[torch.Tensor], lengths: torch.Tensor
    ) -> torch.Tensor | None:
        """Projected frames for each clip, padded to the bucket's length.

        Returns None when the batch is too large for a graph.
        """
        bucket = self._bucket(len(features), max(f.shape[0] for f in features))
        if bucket is None:
            return None
        entry = self.graphs.get(bucket)
        if entry is None:
            entry = self.graphs[bucket] = self._capture(*bucket)
        entry.features.zero_()
        for i, clip in enumerate(features):
            entry.features[i, : clip.shape[0]].copy_(clip)
        entry.lengths.zero_()
        entry.lengths[: len(features)].copy_(lengths)
        entry.graph.replay()
        # The next replay overwrites the output.
        return entry.output[: len(features)].clone()

    def _capture(self, batch_size: int, num_frames: int) -> _GraphEntry:
        features = self.features[: batch_size * num_frames].view(
            batch_size, num_frames, self.num_mel_bins
        )
        lengths = self.lengths[:batch_size]
        positions = self.positions[:num_frames]
        lengths.fill_(num_frames)

        def run() -> torch.Tensor:
            mask = positions[None, :] < lengths[:, None]
            hidden = self.encoder(
                input_features=features, attention_mask=mask
            ).last_hidden_state
            return self.projector(hidden)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
            output = run()
        return _GraphEntry(graph, features, lengths, output)
