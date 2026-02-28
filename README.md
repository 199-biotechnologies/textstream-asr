# TextStream

Live speech-to-text streaming on Apple Silicon. Streams microphone audio through Qwen3-ASR and displays results in a browser.

## Install

```bash
pip install textstream
```

Requires Apple Silicon (M1/M2/M3/M4) and Python 3.10+.

## Usage

```bash
textstream                          # Start with Qwen3-ASR 0.6B
textstream --engine qwen-1.7b       # Higher accuracy model
textstream --vad-threshold 0.5      # Stricter voice detection
textstream --no-browser --no-grafana # Headless mode
```

Open http://localhost:7890 to see live transcription. Switch between Qwen3-ASR 0.6B and 1.7B models via the browser dropdown.

Transcripts are saved to `~/Documents/textstream/transcripts/YYYY-MM-DD/`.
