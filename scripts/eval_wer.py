# SPDX-License-Identifier: Apache-2.0
"""Word error rate and throughput through a running `vllm serve`.

    vllm serve nvidia/parakeet-tdt-0.6b-v3
    python scripts/eval_wer.py --dataset librispeech-clean --concurrency 64

Uses the Whisper English normalizer, like the Open ASR Leaderboard and vLLM's
transcription correctness test.
"""

import argparse
import asyncio
import io
import time

import httpx
import soundfile as sf
from datasets import Audio, load_dataset
from jiwer import wer
from transformers import WhisperTokenizer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

DATASETS = {
    "librispeech-clean": (
        "hf-audio/esb-datasets-test-only-sorted",
        "librispeech",
        "test.clean",
    ),
    "librispeech-other": (
        "hf-audio/esb-datasets-test-only-sorted",
        "librispeech",
        "test.other",
    ),
    "earnings22-tiny": (
        "D4nt3/esb-datasets-earnings22-validation-tiny-filtered",
        None,
        "validation",
    ),
}


def load_rows(name: str, limit: int) -> list[tuple[bytes, float, str]]:
    repo, subset, split = DATASETS[name]
    dataset = load_dataset(repo, subset, split=split)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    rows = []
    for sample in dataset:
        audio, sample_rate = sf.read(
            io.BytesIO(sample["audio"]["bytes"]), dtype="float32"
        )
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        rows.append((buffer.getvalue(), len(audio) / sample_rate, sample["text"]))
    return rows


async def transcribe(client, sem, url, model, wav: bytes) -> str:
    async with sem:
        response = await client.post(
            url,
            files={"file": ("audio.wav", wav, "audio/wav")},
            data={"model": model, "language": "en", "temperature": "0"},
        )
        response.raise_for_status()
        return response.json()["text"]


async def run(args, rows) -> list[str]:
    url = f"{args.base_url}/v1/audio/transcriptions"
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=600) as client:
        await transcribe(client, sem, url, args.model, rows[0][0])  # warm up
        return await asyncio.gather(
            *(transcribe(client, sem, url, args.model, wav) for wav, _, _ in rows)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    parser.add_argument(
        "--dataset", choices=sorted(DATASETS), default="librispeech-clean"
    )
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    rows = load_rows(args.dataset, args.limit)
    start = time.perf_counter()
    predictions = asyncio.run(run(args, rows))
    elapsed = time.perf_counter() - start

    spelling = WhisperTokenizer.from_pretrained(
        "openai/whisper-large-v3"
    ).english_spelling_normalizer
    normalizer = EnglishTextNormalizer(spelling)
    references = [normalizer(text) for _, _, text in rows]
    hypotheses = [normalizer(text) for text in predictions]
    kept = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    audio_s = sum(duration for _, duration, _ in rows)
    print(
        f"dataset: {args.dataset}, {len(rows)} clips, {audio_s / 3600:.2f} h of audio"
    )
    print(f"WER: {100 * wer([r for r, _ in kept], [h for _, h in kept]):.2f}")
    print(f"wall time: {elapsed:.1f} s, RTFx: {audio_s / elapsed:.1f}")


if __name__ == "__main__":
    main()
