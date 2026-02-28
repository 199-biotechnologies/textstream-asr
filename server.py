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
SILENCE_RMS = 0.012
DEFAULT_PORT = 7890
DEFAULT_INTERVAL = 2.5  # seconds between streaming updates

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
paused = False


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
SILENCE_STREAK_RESET = 4      # reset state after N consecutive silent intervals
MAX_STREAM_CHUNKS = 200       # reset state after N chunks to prevent decoder drift


def transcription_loop(model, interval):
    import mlx.core as mx

    broadcast({"type": "status", "content": "Listening..."})
    log("Listening - speak into your microphone (streaming mode)")

    from parakeet_mlx.parakeet import DecodingConfig, Beam

    decoding = DecodingConfig(
        decoding=Beam(beam_size=5, duration_reward=0.7),
    )

    def new_stream():
        return model.transcribe_stream(
            context_size=(384, 384),
            depth=4,
            decoding_config=decoding,
        )

    ctx = new_stream()
    transcriber = ctx.__enter__()

    update_count = 0
    session_chunks = 0
    total_audio_sec = 0.0
    total_infer_ms = 0.0
    prev_finalized_count = 0
    prev_text = ""
    silence_streak = 0

    def reset_stream(reason):
        nonlocal transcriber, ctx, session_chunks, prev_finalized_count, prev_text, silence_streak
        log(f"  -- stream reset ({reason}) --")
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass
        ctx = new_stream()
        transcriber = ctx.__enter__()
        session_chunks = 0
        prev_finalized_count = 0
        prev_text = ""
        silence_streak = 0
        return ctx, transcriber

    while running:
        time.sleep(interval)
        if paused:
            drain_buffer()  # discard audio while paused
            continue
        chunk = drain_buffer()
        if chunk is None or len(chunk) < 800:
            silence_streak += 1
            if silence_streak >= SILENCE_STREAK_RESET and session_chunks > 0:
                ctx, transcriber = reset_stream("sustained silence")
            continue

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < SILENCE_RMS:
            silence_streak += 1
            if silence_streak >= SILENCE_STREAK_RESET and session_chunks > 0:
                ctx, transcriber = reset_stream("sustained silence")
            continue

        silence_streak = 0
        session_chunks += 1

        # Periodic reset to prevent decoder state drift
        if session_chunks >= MAX_STREAM_CHUNKS:
            ctx, transcriber = reset_stream(f"drift prevention after {MAX_STREAM_CHUNKS} chunks")

        audio_sec = len(chunk) / SAMPLE_RATE
        audio_mx = mx.array(chunk)

        t0 = time.perf_counter()
        try:
            transcriber.add_audio(audio_mx)
        except Exception as e:
            log(f"streaming error: {e}")
            ctx, transcriber = reset_stream("error recovery")
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000

        update_count += 1
        total_audio_sec += audio_sec
        total_infer_ms += elapsed_ms

        # Extract finalized and draft text
        finalized_tokens = transcriber.finalized_tokens
        draft_tokens = transcriber.draft_tokens

        fin_text = "".join(t.text for t in finalized_tokens).strip()
        draft_text = "".join(t.text for t in draft_tokens).strip()

        full_text = f"{fin_text} {draft_text}".strip()
        if not full_text or full_text == prev_text:
            continue
        prev_text = full_text

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
            f"RTF={rtf:.4f} | s{session_chunks}/{MAX_STREAM_CHUNKS}] "
            f"{fin_text[-50:]}|{draft_text[:35]}"
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
  display:flex;flex-direction:column;
}
.topbar{
  display:flex;align-items:center;justify-content:space-between;
  padding:0.8rem 1.5rem;
  background:#111;
  border-bottom:1px solid #1a1a1a;
  flex-shrink:0;
  user-select:none;
}
.topbar .left{display:flex;align-items:center;gap:0.6rem}
.topbar .title{font-size:0.7rem;color:#555;letter-spacing:0.12em;text-transform:uppercase;font-weight:600}
.indicator{width:6px;height:6px;border-radius:50%;background:#34d399;transition:background 0.3s}
.indicator.off{background:#555;animation:none}
.indicator.on{animation:pulse 2.5s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:0.3}50%{opacity:1}}
.topbar .controls{display:flex;gap:0.5rem}
.btn{
  background:#1a1a1a;border:1px solid #2a2a2a;color:#888;
  padding:0.35rem 0.9rem;border-radius:4px;font-size:0.65rem;
  cursor:pointer;letter-spacing:0.06em;text-transform:uppercase;
  font-weight:500;transition:all 0.2s;
}
.btn:hover{background:#222;color:#ccc;border-color:#444}
.btn.active{background:#1a2e1a;border-color:#2a4a2a;color:#4ade80}
.btn.danger:hover{background:#2e1a1a;border-color:#4a2a2a;color:#f87171}
.content{
  flex:1;display:flex;align-items:flex-end;
  overflow:hidden;
}
.wrap{
  width:100%;max-width:900px;
  padding:0 2.5rem 3rem;
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
#text .draft{color:#555}
#status{font-size:1rem;color:#444;font-weight:300;padding-bottom:0.8rem}
</style>
</head>
<body>
<div class="topbar">
  <div class="left">
    <div class="indicator on" id="ind"></div>
    <span class="title">textstream</span>
  </div>
  <div class="controls">
    <button class="btn active" id="toggleBtn" onclick="togglePause()">Listening</button>
    <button class="btn danger" onclick="stopServer()">Stop</button>
  </div>
</div>
<div class="content">
  <div class="wrap">
    <div id="status">Loading model...</div>
    <div id="text"></div>
  </div>
</div>
<script>
const textEl=document.getElementById('text');
const statusEl=document.getElementById('status');
const toggleBtn=document.getElementById('toggleBtn');
const ind=document.getElementById('ind');
let isPaused=false;
const MAX_CHARS=500;

function togglePause(){
  isPaused=!isPaused;
  fetch(isPaused?'/pause':'/resume');
  toggleBtn.textContent=isPaused?'Paused':'Listening';
  toggleBtn.classList.toggle('active',!isPaused);
  ind.classList.toggle('on',!isPaused);
  ind.classList.toggle('off',isPaused);
}
function stopServer(){
  if(confirm('Stop TextStream?')){
    fetch('/stop');
    document.body.innerHTML='<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#444;font-size:1.2rem">TextStream stopped.</div>';
  }
}

const src=new EventSource('/stream');
src.onmessage=e=>{
  const d=JSON.parse(e.data);
  if(d.type==='stream'){
    statusEl.style.display='none';
    const fin=d.finalized||'';
    const draft=d.draft||'';
    let visFin=fin;
    if(visFin.length>MAX_CHARS){
      const cut=visFin.indexOf(' ',visFin.length-MAX_CHARS);
      visFin=cut>0?visFin.slice(cut+1):visFin.slice(-MAX_CHARS);
    }
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

        elif self.path == "/pause":
            global paused
            paused = True
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"paused")
            log("Paused")

        elif self.path == "/resume":
            paused = False
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"resumed")
            log("Resumed")

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
