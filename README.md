# Parakeet TDT plugin for vLLM

Out-of-tree vLLM model plugin that serves NVIDIA Parakeet TDT
([`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3))
through vLLM's OpenAI-compatible `/v1/audio/transcriptions` endpoint.

Parakeet TDT is a transducer: a FastConformer encoder, an LSTM prediction
network and a joint network that predicts a token and how many encoder frames
to skip. It does not decode like a Transformer decoder, so it ships as a
plugin instead of an in-tree vLLM model (see vllm-project/vllm#23943).

## Install

```bash
pip install "vllm[audio]>=0.32.0"
pip install git+https://github.com/Alex-Wengg/vllm-parakeet-plugin
```

vLLM discovers the plugin through the `vllm.general_plugins` entry point. If
you use `VLLM_PLUGINS` as an allowlist, add `parakeet` to it.

## Serve

```bash
vllm serve nvidia/parakeet-tdt-0.6b-v3
```

```bash
curl http://localhost:8000/v1/audio/transcriptions \
    -F model=nvidia/parakeet-tdt-0.6b-v3 \
    -F file=@speech.wav
```

The model detects the language itself (25 European languages); `language` is
only validated. Translation, text prompts and logprobs are not supported.

Audio longer than 120 s is split into chunks by the transcription endpoint.
To change the chunk length, pass
`--hf-overrides '{"max_audio_clip_s": 300}'`. Each running request keeps its
encoder frames in GPU memory (about 1 MB per minute of audio), so
`--max-num-seqs` bounds that memory.

## Offline

```python
import soundfile as sf
from vllm import LLM, SamplingParams
from vllm.inputs import ExplicitEncoderDecoderPrompt, TextPrompt, TokensPrompt

llm = LLM(model="nvidia/parakeet-tdt-0.6b-v3")
audio, sample_rate = sf.read("speech.wav", dtype="float32")  # 16 kHz mono
prompt = ExplicitEncoderDecoderPrompt(
    encoder_prompt=TextPrompt(
        prompt="", multi_modal_data={"audio": (audio, sample_rate)}
    ),
    decoder_prompt=TokensPrompt(
        prompt_token_ids=[llm.llm_engine.model_config.hf_config.blank_token_id]
    ),
)
output = llm.generate(prompt, SamplingParams(temperature=0.0, max_tokens=1024))
print(output[0].outputs[0].text)
```

## How it works

- The encoder, prediction network and joint network are the Transformers
  modules (`ParakeetEncoder`, `ParakeetRNNTDecoder`,
  `ParakeetTDTJointNetwork`) loaded from the native checkpoint. The
  prediction and joint networks run in float32.
- Each request owns a slot in a Model Runner V2 `ModelState`: a copy of its
  encoder frames, a frame pointer and the LSTM state.
- On every engine step, `forward` advances all scheduled slots together until
  each one emits a non-blank token, consuming blanks and TDT durations inside
  the loop (`vllm_parakeet_plugin/tdt.py`). `compute_logits` returns one-hot
  logits for that token, so the engine's sampler, stop handling and streaming
  work unchanged. A slot that runs out of frames emits `<|endoftext|>`.
- The loop follows Transformers' `ParakeetTDTGenerationMixin`, plus NeMo's
  `max_symbols_per_step` guard.
- The decode loop is data-dependent, so the plugin runs with `enforce_eager`.

## Tests

```bash
pip install -e ".[test]"
pytest tests/test_tdt.py   # CPU: batched loop vs Transformers generate
python scripts/parity.py   # GPU: vLLM vs Transformers, token by token
python scripts/eval_wer.py --dataset librispeech-clean  # against `vllm serve`
```
