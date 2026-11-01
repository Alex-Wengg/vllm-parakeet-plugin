# SPDX-License-Identifier: Apache-2.0
"""Compare the plugin's greedy tokens with Transformers `generate`.

    python scripts/parity.py --dtype float32 --num-clips 73

Runs both on the same LibriSpeech clips and reports how many transcripts are
token-identical. Needs a CUDA GPU.
"""

import argparse
import io

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, ParakeetForTDT
from vllm import LLM, SamplingParams
from vllm.inputs import ExplicitEncoderDecoderPrompt, TextPrompt, TokensPrompt


def load_clips(num_clips: int) -> list[np.ndarray]:
    import pyarrow.parquet as pq

    path = hf_hub_download(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean/validation-00000-of-00001.parquet",
        repo_type="dataset",
    )
    rows = pq.read_table(path).column("audio").to_pylist()[:num_clips]
    return [sf.read(io.BytesIO(r["bytes"]), dtype="float32")[0] for r in rows]


def reference_tokens(model_id: str, clips: list[np.ndarray]) -> list[list[int]]:
    processor = AutoProcessor.from_pretrained(model_id)
    model = ParakeetForTDT.from_pretrained(model_id, dtype=torch.float32).cuda().eval()
    blank = model.config.blank_token_id
    eos = model.generation_config.eos_token_id
    outputs = []
    for clip in clips:
        inputs = processor(clip, sampling_rate=16_000, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            sequences = model.generate(**inputs).sequences[0].tolist()
        outputs.append([t for t in sequences[1:] if t not in (blank, eos)])
    del model
    torch.cuda.empty_cache()
    return outputs


def plugin_tokens(
    model_id: str,
    clips: list[np.ndarray],
    dtype: str,
    max_num_seqs: int,
    enforce_eager: bool,
) -> list[list[int]]:
    llm = LLM(
        model=model_id,
        dtype=dtype,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=0.3,
        enforce_eager=enforce_eager,
    )
    blank = llm.llm_engine.model_config.hf_config.blank_token_id
    eos = llm.llm_engine.model_config.hf_config.eos_token_id
    prompts = [
        ExplicitEncoderDecoderPrompt(
            encoder_prompt=TextPrompt(
                prompt="", multi_modal_data={"audio": (clip, 16_000)}
            ),
            decoder_prompt=TokensPrompt(prompt_token_ids=[blank]),
        )
        for clip in clips
    ]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1024))
    return [[t for t in o.outputs[0].token_ids if t != eos] for o in outputs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--num-clips", type=int, default=73)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument(
        "--enforce-eager", action="store_true", help="turn off CUDA graphs"
    )
    args = parser.parse_args()

    clips = load_clips(args.num_clips)
    expected = reference_tokens(args.model, clips)
    actual = plugin_tokens(
        args.model, clips, args.dtype, args.max_num_seqs, args.enforce_eager
    )

    processor = AutoProcessor.from_pretrained(args.model)
    matches = 0
    for i, (exp, act) in enumerate(zip(expected, actual)):
        if exp == act:
            matches += 1
            continue
        print(f"clip {i} differs")
        print("  transformers:", processor.decode(exp))
        print("  vllm:        ", processor.decode(act))
    mode = "eager" if args.enforce_eager else "CUDA graphs"
    print(f"{matches}/{len(clips)} transcripts token-identical ({args.dtype}, {mode})")


if __name__ == "__main__":
    main()
