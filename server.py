#!/usr/bin/env python3
"""TextStream — Live speech-to-text streaming via Parakeet ASR.

Uses parakeet-mlx's native streaming API for near word-by-word transcription.
Streams results to a browser (SSE) and Grafana Cloud (annotations).
Saves daily transcripts to documents/transcripts/YYYY-MM-DD/.

Usage:
    python server.py                # Start with defaults (1.5s chunks, English)
    python server.py --port 8080    # Custom port
    python server.py --model v3     # Multilingual (25 languages)
    python server.py --interval 1   # Faster updates (seconds between add_audio calls)
"""

import sys
import os
import json
import time
import signal
import threading
import queue
import webbrowser
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from socketserver import ThreadingMixIn
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import sounddevice as sd

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
SILENCE_RMS = 0.006
DEFAULT_PORT = 7890
DEFAULT_INTERVAL = 1.5  # seconds between streaming updates

GRAFANA_URL = os.environ.get("GRAFANA_URL", "https://triscient.grafana.net")
GRAFANA_TOKEN = os.environ.get(
    "GRAFANA_SERVICE_ACCOUNT_TOKEN",
    "glsa_Ug6swwrUH57IzEjfuK3d0S4XrBm36R1y_d7ef78e8",
)

# ── State ─────────────────────────────────────────────────────────────────────
audio_chunks = []
buffer_lock = threading.Lock()
subscribers = []
sub_lock = threading.Lock()
running = True


def log(msg):
    print(f"\033[90m[textstream]\033[0m {msg}", file=sys.stderr, flush=True)


# ── Audio Capture ─────────────────────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    if status:
        log(f"audio warning: {status}")
    with buffer_lock:
        audio_chunks.append(indata[:, 0].copy())


def drain_buffer():
    global audio_chunks
    with buffer_lock:
        if not audio_chunks:
            return None
        chunk = np.concatenate(audio_chunks)
        audio_chunks = []
    return chunk


# ── SSE Broadcast ─────────────────────────────────────────────────────────────
def broadcast(event_data):
    payload = f"data: {json.dumps(event_data)}\n\n".encode()
    with sub_lock:
        dead = []
        for q in subscribers:
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for d in dead:
            subscribers.remove(d)


# ── Grafana Annotation Push ───────────────────────────────────────────────────
def push_annotation(text):
    try:
        payload = json.dumps({
            "text": text,
            "tags": ["textstream"],
            "time": int(time.time() * 1000),
        }).encode()
        req = urllib.request.Request(
            f"{GRAFANA_URL}/api/annotations",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GRAFANA_TOKEN}",
                "User-Agent": "textstream/1.0",
            },
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        log(f"grafana push failed: {e}")


# ── Transcript File Persistence ───────────────────────────────────────────────
TRANSCRIPT_DIR = Path(__file__).parent / "documents" / "transcripts"
_transcript_file = None
_transcript_lock = threading.Lock()


def get_transcript_file():
    global _transcript_file
    from datetime import datetime

    now = datetime.now()
    day_dir = TRANSCRIPT_DIR / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    if _transcript_file is None or _transcript_file.closed:
        session_name = now.strftime("%H-%M-%S")
        path = day_dir / f"{session_name}.txt"
        _transcript_file = open(path, "a", encoding="utf-8")
        log(f"Transcript: {path}")
    return _transcript_file


def save_transcript(text):
    from datetime import datetime

    with _transcript_lock:
        f = get_transcript_file()
        ts = datetime.now().strftime("%H:%M:%S")
        f.write(f"[{ts}] {text}\n")
        f.flush()


# ── Streaming Transcription Loop ─────────────────────────────────────────────
def transcription_loop(model, interval):
    import mlx.core as mx

    broadcast({"type": "status", "content": "Listening..."})
    log("Listening - speak into your microphone (streaming mode)")

    update_count = 0
    total_audio_sec = 0.0
    total_infer_ms = 0.0
    prev_finalized_count = 0
    prev_text = ""

    with model.transcribe_stream(context_size=(256, 256), depth=1) as transcriber:
        while running:
            time.sleep(interval)
            chunk = drain_buffer()
            if chunk is None or len(chunk) < 800:  # < 50ms, skip
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < SILENCE_RMS:
                continue

            audio_sec = len(chunk) / SAMPLE_RATE
            audio_mx = mx.array(chunk)

            t0 = time.perf_counter()
            try:
                transcriber.add_audio(audio_mx)
            except Exception as e:
                log(f"streaming error: {e}")
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000

            update_count += 1
            total_audio_sec += audio_sec
            total_infer_ms += elapsed_ms

            # Extract finalized and draft text
            finalized_tokens = transcriber.finalized_tokens
            draft_tokens = transcriber.draft_tokens

            fin_text = "".join(
                t.text for t in finalized_tokens
            ).strip()
            draft_text = "".join(
                t.text for t in draft_tokens
            ).strip()

            full_text = f"{fin_text} {draft_text}".strip()
            if not full_text or full_text == prev_text:
                continue

            prev_text = full_text

            # Broadcast update with both finalized and draft
            broadcast({
                "type": "stream",
                "finalized": fin_text,
                "draft": draft_text,
            })

            # Save newly finalized text to disk + Grafana
            new_fin_count = len(finalized_tokens)
            if new_fin_count > prev_finalized_count:
                new_text = "".join(
                    t.text for t in finalized_tokens[prev_finalized_count:]
                ).strip()
                if new_text:
                    save_transcript(new_text)
                    threading.Thread(
                        target=push_annotation, args=(new_text,), daemon=True
                    ).start()
                prev_finalized_count = new_fin_count

            avg_ms = total_infer_ms / update_count
            rtf = (total_infer_ms / 1000) / total_audio_sec if total_audio_sec > 0 else 0
            log(
                f"  [{elapsed_ms:.0f}ms | avg {avg_ms:.0f}ms | "
                f"RTF={rtf:.4f} | #{update_count}] "
                f"{fin_text[-60:]}|{draft_text[:40]}"
            )


# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TextStream</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden}
body{
  background:#09090b;
  color:#e8e8e8;
  font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text','Inter',sans-serif;
  display:flex;align-items:flex-end;justify-content:flex-start;
}
.wrap{
  width:100%;max-width:900px;
  padding:0 3rem 5rem;
  margin:0 auto;
}
#text{
  font-size:1.65rem;
  font-weight:350;
  line-height:1.7;
  letter-spacing:-0.01em;
  text-align:left;
  word-wrap:break-word;
}
#text .draft{
  color:#666;
}
#status{
  font-size:1.1rem;
  color:#444;
  font-weight:300;
  padding-bottom:1rem;
}
.bar{
  position:fixed;bottom:1.4rem;right:2rem;
  display:flex;align-items:center;gap:0.4rem;
  font-size:0.55rem;color:#2a2a2a;letter-spacing:0.12em;text-transform:uppercase;
  font-weight:500;
}
.dot{
  width:4px;height:4px;border-radius:50%;
  background:#34d399;
  animation:pulse 3s ease-in-out infinite;
}
@keyframes pulse{0%,100%{opacity:0.2}50%{opacity:1}}
</style>
</head>
<body>
<div class="wrap">
  <div id="status">Loading model...</div>
  <div id="text"></div>
</div>
<div class="bar"><div class="dot"></div><span>textstream</span></div>
<script>
const textEl=document.getElementById('text');
const statusEl=document.getElementById('status');
let shownFin=0;
const MAX_CHARS=400;

const src=new EventSource('/stream');
src.onmessage=e=>{
  const d=JSON.parse(e.data);

  if(d.type==='stream'){
    statusEl.style.display='none';
    const fin=d.finalized||'';
    const draft=d.draft||'';

    // Build visible text: tail of finalized + draft
    // Only show last MAX_CHARS of finalized for performance
    let visFin=fin;
    if(visFin.length>MAX_CHARS){
      // Cut at a space boundary
      const cut=visFin.indexOf(' ',visFin.length-MAX_CHARS);
      visFin=cut>0?visFin.slice(cut+1):visFin.slice(-MAX_CHARS);
    }

    // Set innerHTML once: finalized as plain text, draft in span
    if(draft){
      textEl.innerHTML=esc(visFin)+'<span class="draft"> '+esc(draft)+'</span>';
    }else{
      textEl.textContent=visFin;
    }

  }else if(d.type==='status'){
    statusEl.style.display='block';
    statusEl.textContent=d.content;
  }
};
src.onerror=()=>{
  statusEl.style.display='block';
  statusEl.textContent='Reconnecting...';
};
function esc(s){
  const d=document.createElement('div');
  d.textContent=s;
  return d.innerHTML;
}
</script>
</body>
</html>"""


# ── HTTP Server ───────────────────────────────────────────────────────────────
class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode())

        elif self.path == "/stop":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"TextStream stopped.\n")
            log("Stop requested via /stop endpoint")
            threading.Thread(target=lambda: os.kill(os.getpid(), signal.SIGTERM), daemon=True).start()

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            q = queue.Queue()
            with sub_lock:
                subscribers.append(q)
            try:
                while running:
                    try:
                        data = q.get(timeout=15)
                        self.wfile.write(data)
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with sub_lock:
                    if q in subscribers:
                        subscribers.remove(q)

        else:
            self.send_error(404)

    def log_message(self, *_):
        pass


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global running

    parser = argparse.ArgumentParser(description="TextStream - live speech to text")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--model", default="v2", choices=["v2", "v3"],
        help="v2=English, v3=25 languages (default: v2)",
    )
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL,
        help="Seconds between streaming updates (default: 1.5, min safe: 1.0)",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-grafana", action="store_true", help="Disable Grafana push")
    args = parser.parse_args()

    if args.no_grafana:
        global push_annotation
        push_annotation = lambda text: None

    model_map = {
        "v2": "mlx-community/parakeet-tdt-0.6b-v2",
        "v3": "mlx-community/parakeet-tdt-0.6b-v3",
    }

    log(f"Loading Parakeet {args.model}...")
    from parakeet_mlx import from_pretrained
    model = from_pretrained(model_map[args.model])
    log("Model ready")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1),
    )
    stream.start()
    log("Microphone active")

    t = threading.Thread(
        target=transcription_loop, args=(model, args.interval), daemon=True
    )
    t.start()

    httpd = ThreadedServer(("127.0.0.1", args.port), Handler)

    url = f"http://localhost:{args.port}"
    log(f"Browser:  {url}")
    log(f"Grafana:  annotations tagged 'textstream'")
    log(f"Interval: {args.interval}s streaming updates")

    if not args.no_browser:
        webbrowser.open(url)

    def shutdown(sig, frame):
        global running
        running = False
        log("\nStopping...")
        stream.stop()
        stream.close()
        httpd.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
