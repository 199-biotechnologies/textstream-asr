# TextStream

<p align="center">
  <img src="logo.png" alt="TextStream" width="600">
</p>

<p align="center">
  <strong>Local real-time speech-to-text for Apple Silicon. One pip install. No API keys. No cloud. No cost.</strong>
</p>

<p align="center">
  <code>pip install textstream-asr</code>&nbsp;&nbsp;&nbsp;then&nbsp;&nbsp;&nbsp;<code>textstream</code>
</p>

---

TextStream turns your Mac's microphone into a live transcription server. It runs Qwen3-ASR (~2% word error rate) on-device through MLX, filters noise with Silero VAD, and streams text over SSE at `localhost:7890/stream`. Any app, script, or frontend can subscribe and get words as they're spoken.

Build voice-controlled tools. Add live captions to your app. Record meeting notes that write themselves. Pipe speech into your IDE. Whatever needs ears — point it at the stream.

### Why this exists

Cloud speech APIs charge per minute and add latency. Whisper runs offline but isn't real-time. TextStream gives you a live, local transcription endpoint that any process on your machine can read from — for free, with 2% WER accuracy.

## Benchmarks

Numbers from published evaluations. Your actual RTF will depend on model size and what else is running.

### Accuracy (Word Error Rate)

| Model | LibriSpeech clean | LibriSpeech other | Params |
|-------|:-----------------:|:-----------------:|:------:|
| **Qwen3-ASR 0.6B** (default) | 2.11% | 4.55% | 600M |
| **Qwen3-ASR 1.7B** | 1.63% | 3.38% | 1.7B |
| Whisper-large-v3 | 1.51% | 3.97% | 1.5B |
| GPT-4o-Transcribe | 1.39% | 3.75% | — |

Source: [Qwen3-ASR Technical Report](https://arxiv.org/abs/2601.21337)

### Speed (Apple Silicon via MLX)

| Metric | Value |
|--------|-------|
| Real-time factor (RTF) | ~0.06 (16x faster than real-time) |
| MLX vs PyTorch | ~4x faster on Apple Silicon |
| VAD latency | <1ms per 32ms audio chunk |
| Time to first token | ~92ms |

Source: [mlx-qwen3-asr benchmarks](https://github.com/moona3k/mlx-qwen3-asr), [Silero VAD performance metrics](https://github.com/snakers4/silero-vad/wiki/Performance-Metrics)

### Resource usage

- **RAM**: ~1.2GB for 0.6B model, ~3GB for 1.7B
- **CPU/GPU**: Runs on Neural Engine + GPU via MLX Metal backend. Minimal CPU overhead — the transcription loop sleeps between intervals
- **Disk**: Models are cached by HuggingFace Hub (~1.2GB / 3.4GB first download)
- **Battery**: Comparable to background music playback. MLX is designed for Apple Silicon power efficiency

## Requirements

| | Supported |
|--|-----------|
| **macOS** on Apple Silicon (M1/M2/M3/M4) | Yes |
| macOS on Intel | No — MLX requires Apple Silicon |
| Linux / Windows | Not yet — MLX is macOS-only. PyTorch backend planned |
| Python | 3.10+ |

## Install

```bash
pip install textstream-asr
```

## Quick start

```bash
textstream                            # start transcribing, opens browser UI
textstream --no-browser               # headless — just the SSE server
textstream --engine qwen-1.7b         # larger model, lower word error rate
textstream --vad-threshold 0.5        # stricter voice detection (default 0.4)
```

### Connect from your app

```python
import json, urllib.request

# Subscribe to the live transcript stream
req = urllib.request.Request("http://localhost:7890/stream")
with urllib.request.urlopen(req) as resp:
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event["type"] == "stream":
                print(event["finalized"], event["draft"])
```

```javascript
// Browser / Node SSE
const src = new EventSource("http://localhost:7890/stream");
src.onmessage = (e) => {
  const { finalized, draft } = JSON.parse(e.data);
  console.log(finalized, draft);
};
```

## How it works

Every `--interval` seconds (default 2.5), TextStream drains the mic buffer and runs Silero VAD on the chunk. If speech is detected, the chunk is fed to Qwen3-ASR's streaming decoder. The model returns stable (finalized) text and speculative (draft) text. Stable text gets persisted to disk and broadcast to all SSE subscribers.

If the model hallucinates on noise that slips past VAD, a pattern filter catches it and resets the stream. Safety net — with VAD active, it almost never fires.

## API

```
GET /stream    → SSE stream: {"type":"stream","finalized":"...","draft":"..."}
GET /engine    → {"engine":"qwen"}
GET /switch?engine=qwen-1.7b → hot-swap model without restart
GET /pause     → pause mic capture
GET /resume    → resume
GET /stop      → shutdown
GET /          → built-in browser UI
```

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 7890 | HTTP server port |
| `--engine` | qwen | `qwen` (0.6B) or `qwen-1.7b` |
| `--interval` | 2.5 | Seconds between transcription updates |
| `--vad-threshold` | 0.4 | Silero VAD speech probability threshold |
| `--no-browser` | — | Don't open browser on start |

Transcripts are saved to `~/Documents/textstream/transcripts/YYYY-MM-DD/`.

## Dependencies

- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework
- [mlx-qwen3-asr](https://github.com/niclas3640/mlx-qwen3-asr) — Qwen3-ASR for MLX
- [silero-vad-lite](https://github.com/snakers4/silero-vad-lite) — Voice activity detection (~2MB, bundles ONNX runtime)
- [sounddevice](https://python-sounddevice.readthedocs.io/) — PortAudio bindings
- NumPy

## Author

Boris Djordjevic — [199 Biotechnologies](https://github.com/199-biotechnologies)

## License

MIT
