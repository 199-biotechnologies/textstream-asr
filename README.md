<p align="center">
  <img src="logo.png" alt="TextStream — Live Speech-to-Text on Apple Silicon" width="600">
</p>

<h1 align="center">TextStream</h1>

<p align="center">
  <strong>Live speech-to-text streaming on Apple Silicon. One command. No API keys. No cloud.</strong>
</p>

<p align="center">
  <a href="https://github.com/199-biotechnologies/textstream-asr/stargazers">
    <img src="https://img.shields.io/github/stars/199-biotechnologies/textstream-asr?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow" alt="Star this repo">
  </a>
  &nbsp;
  <a href="https://x.com/longevityboris">
    <img src="https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow @longevityboris">
  </a>
</p>

<p align="center">
  <a href="https://pypi.org/project/textstream-asr/">
    <img src="https://img.shields.io/pypi/v/textstream-asr?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI" alt="PyPI version">
  </a>
  &nbsp;
  <a href="https://github.com/199-biotechnologies/textstream-asr/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License">
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  &nbsp;
  <img src="https://img.shields.io/badge/Apple_Silicon-M1_|_M2_|_M3_|_M4-000000?style=for-the-badge&logo=apple&logoColor=white" alt="Apple Silicon">
  &nbsp;
  <img src="https://img.shields.io/badge/MLX-Native-ff6600?style=for-the-badge" alt="MLX Native">
</p>

<br>

TextStream turns your Mac's microphone into a live transcription endpoint. It runs [Qwen3-ASR](https://arxiv.org/abs/2601.21337) on-device through [MLX](https://github.com/ml-explore/mlx), filters noise with [Silero VAD](https://github.com/snakers4/silero-vad), and streams text over SSE at `localhost:7890/stream`. Any app, script, or frontend can subscribe and get words as they are spoken — with ~2% word error rate, no API keys, and zero cost.

<p align="center">
  <a href="#install">Install</a> · <a href="#quick-start">Quick Start</a> · <a href="#how-it-works">How It Works</a> · <a href="#api">API</a> · <a href="#benchmarks">Benchmarks</a> · <a href="#configuration">Configuration</a> · <a href="#contributing">Contributing</a>
</p>

---

## Why This Exists

Cloud speech APIs charge per minute and add network latency. Whisper runs offline but is not real-time. There is no simple way to get a local, streaming transcription endpoint that any process on your machine can read from.

TextStream fills that gap. One `pip install`, one command, and every app on your machine has access to a live transcript stream — for free.

Build voice-controlled tools. Add live captions to your app. Record meeting notes that write themselves. Pipe speech into your IDE. Whatever needs ears, point it at the stream.

## Install

```bash
pip install textstream-asr
```

**Requirements:** macOS on Apple Silicon (M1/M2/M3/M4), Python 3.10+.

## Quick Start

```bash
textstream                            # start transcribing, opens browser UI
textstream --no-browser               # headless — SSE server only
textstream --engine qwen-1.7b         # larger model, lower word error rate
textstream --vad-threshold 0.5        # stricter voice detection (default 0.4)
```

### Connect from your app

**Python:**

```python
import json, urllib.request

req = urllib.request.Request("http://localhost:7890/stream")
with urllib.request.urlopen(req) as resp:
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event["type"] == "stream":
                print(event["finalized"], event["draft"])
```

**JavaScript:**

```javascript
const src = new EventSource("http://localhost:7890/stream");
src.onmessage = (e) => {
  const { finalized, draft } = JSON.parse(e.data);
  console.log(finalized, draft);
};
```

## How It Works

Every `--interval` seconds (default 2.5), TextStream drains the mic buffer and runs Silero VAD on the chunk. If speech is detected, the chunk goes to Qwen3-ASR's streaming decoder. The model returns stable (finalized) text and speculative (draft) text. Stable text gets persisted to disk and broadcast to all SSE subscribers.

If the model hallucinates on noise that slips past VAD, a pattern filter catches it and resets the stream. With VAD active, this almost never fires.

```
Microphone → Audio Buffer → Silero VAD → Qwen3-ASR (MLX) → SSE Stream
                              ↓ (no speech)
                            Skip chunk
```

## API

| Endpoint | Description |
|----------|-------------|
| `GET /stream` | SSE stream: `{"type":"stream","finalized":"...","draft":"..."}` |
| `GET /engine` | Current engine info |
| `GET /switch?engine=qwen-1.7b` | Hot-swap model without restart |
| `GET /pause` | Pause mic capture |
| `GET /resume` | Resume mic capture |
| `GET /stop` | Shutdown server |
| `GET /` | Built-in browser UI |

## Benchmarks

### Accuracy (Word Error Rate)

| Model | LibriSpeech clean | LibriSpeech other | Params |
|-------|:-----------------:|:-----------------:|:------:|
| **Qwen3-ASR 0.6B** (default) | 2.11% | 4.55% | 600M |
| **Qwen3-ASR 1.7B** | 1.63% | 3.38% | 1.7B |
| Whisper-large-v3 | 1.51% | 3.97% | 1.5B |
| GPT-4o-Transcribe | 1.39% | 3.75% | -- |

Source: [Qwen3-ASR Technical Report](https://arxiv.org/abs/2601.21337)

### Speed (Apple Silicon via MLX)

| Metric | Value |
|--------|-------|
| Real-time factor (RTF) | ~0.06 (16x faster than real-time) |
| MLX vs PyTorch | ~4x faster on Apple Silicon |
| VAD latency | <1ms per 32ms audio chunk |
| Time to first token | ~92ms |

Source: [mlx-qwen3-asr benchmarks](https://github.com/moona3k/mlx-qwen3-asr), [Silero VAD metrics](https://github.com/snakers4/silero-vad/wiki/Performance-Metrics)

### Resource Usage

- **RAM:** ~1.2 GB for 0.6B model, ~3 GB for 1.7B
- **CPU/GPU:** Runs on Neural Engine + GPU via MLX Metal backend. Minimal CPU overhead.
- **Disk:** Models cached by HuggingFace Hub (~1.2 GB / 3.4 GB first download)
- **Battery:** Comparable to background music playback. MLX is designed for Apple Silicon power efficiency.

## Features

- **Real-time streaming ASR** via Server-Sent Events at `localhost:7890/stream`
- **Qwen3-ASR on MLX** — 2% WER, 16x faster than real-time on Apple Silicon
- **Silero VAD** filters silence and noise before transcription runs
- **Hot-swap models** between 0.6B and 1.7B without restarting the server
- **Built-in browser UI** for quick visual monitoring
- **Hallucination filter** catches and resets repetitive model output
- **Auto-saves transcripts** to `~/Documents/textstream/transcripts/`
- **Zero dependencies on cloud services** — runs entirely on-device

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `7890` | HTTP server port |
| `--engine` | `qwen` | `qwen` (0.6B) or `qwen-1.7b` |
| `--interval` | `2.5` | Seconds between transcription updates |
| `--vad-threshold` | `0.4` | Silero VAD speech probability threshold |
| `--no-browser` | -- | Do not open browser on start |

Transcripts are saved to `~/Documents/textstream/transcripts/YYYY-MM-DD/`.

## Dependencies

- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework for Apple Silicon
- [mlx-qwen3-asr](https://github.com/niclas3640/mlx-qwen3-asr) — Qwen3-ASR ported to MLX
- [silero-vad-lite](https://github.com/snakers4/silero-vad-lite) — Voice activity detection (~2 MB, bundles ONNX runtime)
- [sounddevice](https://python-sounddevice.readthedocs.io/) — PortAudio bindings for mic capture
- [NumPy](https://numpy.org/)

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)

---

<p align="center">
  Built by <a href="https://github.com/longevityboris">Boris Djordjevic</a> at <a href="https://github.com/199-biotechnologies">199 Biotechnologies</a> | <a href="https://paperfoot.ai">Paperfoot AI</a>
</p>

<p align="center">
  <a href="https://github.com/199-biotechnologies/textstream-asr/stargazers">
    <img src="https://img.shields.io/github/stars/199-biotechnologies/textstream-asr?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow" alt="Star this repo">
  </a>
  &nbsp;
  <a href="https://x.com/longevityboris">
    <img src="https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow @longevityboris">
  </a>
</p>
