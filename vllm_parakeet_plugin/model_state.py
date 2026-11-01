# SPDX-License-Identifier: Apache-2.0
"""Model Runner V2 state: one TDT decoding slot per request."""

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from .tdt import TDTSlots


class ParakeetTDTModelState(ModelState):
    """Keeps each request's acoustic frames and LSTM state in its slot."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        assert encoder_cache is not None
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model.config
        budget = MultiModalBudget(vllm_config, MULTIMODAL_REGISTRY, enable_cache=False)
        # The shared encoder cache drops a request's frames after prefill, but
        # TDT reads them until the transcript ends, so each slot keeps a copy.
        self.tdt = TDTSlots(
            num_slots=self.max_num_reqs,
            max_frames=budget.mm_max_toks_per_item["audio"],
            hidden_size=config.decoder_hidden_size,
            num_layers=config.num_decoder_layers,
            frames_dtype=self.dtype,
            dtype=torch.float32,
            device=device,
        )
        self.audio_hashes: dict[str, str] = {}
        self.slots: dict[str, int] = {}
        self.num_tokens_to_replay: dict[str, int] = {}
        self.loaded: set[str] = set()

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        audio = [f for f in new_req_data.mm_features if f.modality == "audio"]
        if not audio and new_req_data.req_id.startswith("_warmup_"):
            return
        if len(audio) != 1:
            raise ValueError("Parakeet TDT expects one audio per request.")
        req_id = new_req_data.req_id
        self.audio_hashes[req_id] = audio[0].identifier
        self.slots[req_id] = req_index
        # A resumed request arrives with its earlier output in the prompt.
        assert new_req_data.prefill_token_ids is not None
        self.num_tokens_to_replay[req_id] = (
            len(new_req_data.prefill_token_ids) - new_req_data.prompt_len
        )

    def remove_request(self, req_id: str) -> None:
        self.audio_hashes.pop(req_id, None)
        self.slots.pop(req_id, None)
        self.num_tokens_to_replay.pop(req_id, None)
        self.loaded.discard(req_id)

    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> None:
        self.execute_mm_encoder(scheduled_encoder_inputs)
        return None

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        blank_token_id = self.model.config.blank_token_id
        ready = (
            input_batch.num_computed_tokens_np + input_batch.num_scheduled_tokens
            >= input_batch.prefill_len_np
        )
        query_end = input_batch.query_start_loc_np[1 : input_batch.num_reqs + 1]
        slots: list[int] = []
        rows: list[int] = []
        for i, req_id in enumerate(input_batch.req_ids):
            slot = self.slots.get(req_id)
            if slot is None:
                continue  # Dummy and warmup requests carry no audio.
            if req_id not in self.loaded:
                audio_hash = self.audio_hashes[req_id]
                frames = self.encoder_cache.encoder_outputs.get(audio_hash)
                if frames is None:
                    continue
                self.tdt.reset(
                    slot,
                    frames,
                    blank_token_id=blank_token_id,
                    num_tokens_to_replay=self.num_tokens_to_replay.pop(req_id),
                )
                self.loaded.add(req_id)
            if ready[i]:
                slots.append(slot)
                rows.append(int(query_end[i]) - 1)
        if not slots:
            return {}
        return {
            "tdt_state": self.tdt,
            "tdt_slots": torch.tensor(slots, dtype=torch.long, device=self.device),
            "tdt_rows": torch.tensor(rows, dtype=torch.long, device=self.device),
        }

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        # Dummy runs never touch the live slots.
        return {}

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        return {}  # Parakeet TDT has no attention KV cache.
