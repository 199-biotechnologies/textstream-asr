#!/usr/bin/env python3
"""TextStream — Local real-time speech-to-text server for Apple Silicon.

Captures microphone audio, filters with Silero VAD, transcribes with
Qwen3-ASR on MLX, and streams results over SSE at localhost:7890/stream.
Any app or script can subscribe for near real-time transcription.

Usage:
    textstream                          # Start server, open browser UI
    textstream --no-browser             # Headless — SSE server only
    textstream --engine qwen-1.7b       # Larger model, lower WER
    textstream --vad-threshold 0.5      # Stricter voice activity detection
    textstream --port 8080              # Custom port
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
from abc import ABC, abstractmethod
from pathlib import Path
from socketserver import ThreadingMixIn
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import sounddevice as sd

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
DEFAULT_PORT = 7890
DEFAULT_INTERVAL = 2.5
DEFAULT_VAD_THRESHOLD = 0.4

GRAFANA_URL = os.environ.get("GRAFANA_URL", "https://triscient.grafana.net")
GRAFANA_TOKEN = os.environ.get("GRAFANA_SERVICE_ACCOUNT_TOKEN", "")

# ── State ─────────────────────────────────────────────────────────────────────
audio_queue = queue.Queue(maxsize=int(SAMPLE_RATE * 10 / 1600))  # Cap ~10s of audio
subscribers = []
sub_lock = threading.Lock()
running = True
paused = False
current_engine = None
engine_lock = threading.Lock()
pending_engine_name = None  # Set by /switch, consumed by transcription loop
annotation_queue = queue.Queue(maxsize=100)


def log(msg):
    print(f"\033[90m[textstream]\033[0m {msg}", file=sys.stderr, flush=True)


# ── Audio Capture ─────────────────────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    """PortAudio callback — must be realtime-safe (no locks, no logging)."""
    try:
        audio_queue.put_nowait(indata[:, 0].copy())
    except queue.Full:
        pass  # Drop oldest — inference is lagging


def drain_buffer():
    chunks = []
    try:
        while True:
            chunks.append(audio_queue.get_nowait())
    except queue.Empty:
        pass
    if not chunks:
        return None
    return np.concatenate(chunks)


# ── SSE Broadcast ─────────────────────────────────────────────────────────────
def broadcast(event_data):
    payload = f"data: {json.dumps(event_data)}\n\n".encode()
    with sub_lock:
        dead = []
        for q in subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                # Drop oldest event for slow clients
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except Exception:
                    dead.append(q)
            except Exception:
                dead.append(q)
        for d in dead:
            subscribers.remove(d)


# ── Grafana Annotation Push ───────────────────────────────────────────────────
def _push_annotation_http(text):
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


def _annotation_worker():
    """Single background thread draining annotation_queue."""
    while running:
        try:
            text = annotation_queue.get(timeout=1)
            _push_annotation_http(text)
        except queue.Empty:
            continue
        except Exception:
            continue


def push_annotation(text):
    """Queue an annotation for async push (non-blocking)."""
    try:
        annotation_queue.put_nowait(text)
    except queue.Full:
        pass


# ── Transcript File Persistence ───────────────────────────────────────────────
TRANSCRIPT_DIR = Path.home() / "Documents" / "textstream" / "transcripts"
_transcript_file = None
_transcript_lock = threading.Lock()


def close_transcript():
    with _transcript_lock:
        global _transcript_file
        if _transcript_file and not _transcript_file.closed:
            _transcript_file.flush()
            _transcript_file.close()
            _transcript_file = None


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


# ── Engine Abstraction ────────────────────────────────────────────────────────
class ASREngine(ABC):
    name: str = "unknown"
    description: str = ""

    @abstractmethod
    def start(self):
        """Initialize streaming state."""

    @abstractmethod
    def feed(self, audio_np: np.ndarray) -> tuple[str, str]:
        """Feed audio chunk, return (stable_text, draft_text)."""

    @abstractmethod
    def stop(self):
        """Finalize and clean up streaming state."""

    @abstractmethod
    def needs_manual_reset(self) -> bool:
        """Whether this engine needs periodic drift resets."""

    def reset(self):
        """Reset streaming state (for engines that drift)."""
        self.stop()
        self.start()


class QwenEngine(ASREngine):
    description = "Qwen3-ASR 0.6B — accurate (~2.3% WER), built-in context window"

    # Known hallucination patterns (model regurgitates its chat system prompt on noise)
    HALLUCINATION_PATTERNS = [
        "you are a helpful assistant",
        "i am a helpful assistant",
        "as an ai",
        "as a language model",
    ]

    def __init__(self, model_size="0.6b"):
        self.model_size = model_size
        self.name = "qwen" if model_size == "0.6b" else "qwen-1.7b"
        self.session = None
        self._state = None
        self._prev_stable_len = 0
        self._hallucination_streak = 0

    def load(self):
        model_map = {
            "0.6b": "Qwen/Qwen3-ASR-0.6B",
            "1.7b": "Qwen/Qwen3-ASR-1.7B",
        }
        log(f"Loading Qwen3-ASR {self.model_size}...")
        from mlx_qwen3_asr import Session
        self.session = Session(model=model_map[self.model_size])
        log("Qwen3-ASR ready")

    def start(self):
        if self.session is None:
            self.load()
        self._state = self.session.init_streaming(
            chunk_size_sec=2.0,
            max_context_sec=30.0,
            finalization_mode="accuracy",
            sample_rate=SAMPLE_RATE,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            language="English",
        )
        self._prev_stable_len = 0

    def _is_hallucination(self, text):
        lower = text.lower().strip()
        return any(p in lower for p in self.HALLUCINATION_PATTERNS)

    def feed(self, audio_np):
        self._state = self.session.feed_audio(audio_np, self._state)

        stable = self._state.stable_text.strip()
        full = self._state.text.strip()

        # Draft = everything after stable
        draft = ""
        if len(full) > len(stable):
            draft = full[len(stable):].strip()

        # Detect hallucination — model regurgitates system prompt on noise/music
        if self._is_hallucination(draft) or self._is_hallucination(full[-60:] if full else ""):
            self._hallucination_streak += 1
            if self._hallucination_streak >= 2:
                log("  -- hallucination detected, resetting stream --")
                self.stop()
                self.start()
                return "", ""
            return stable, ""  # suppress the hallucinated draft
        else:
            self._hallucination_streak = 0

        # Track newly finalized text for persistence
        new_text = ""
        if len(stable) > self._prev_stable_len:
            new_text = stable[self._prev_stable_len:].strip()
            # Don't persist hallucinated text
            if new_text and not self._is_hallucination(new_text):
                save_transcript(new_text)
                push_annotation(new_text)
            self._prev_stable_len = len(stable)

        return stable, draft

    def stop(self):
        if self._state and self.session:
            try:
                self._state = self.session.finish_streaming(self._state)
                final = self._state.stable_text.strip()
                if len(final) > self._prev_stable_len:
                    remaining = final[self._prev_stable_len:].strip()
                    if remaining:
                        save_transcript(remaining)
                        push_annotation(remaining)
            except Exception as e:
                log(f"qwen finish error: {e}")
        self._state = None
        self._prev_stable_len = 0

    def needs_manual_reset(self):
        return False  # built-in 30s sliding window handles context


ENGINES = {
    "qwen": lambda: QwenEngine("0.6b"),
    "qwen-1.7b": lambda: QwenEngine("1.7b"),
}


# ── Streaming Transcription Loop ─────────────────────────────────────────────
SILENCE_STREAK_RESET = 4


def transcription_loop(interval, vad_threshold):
    global current_engine, pending_engine_name

    import gc

    from .vad import contains_speech

    broadcast({"type": "status", "content": "Listening..."})
    log("Listening - speak into your microphone (streaming mode)")

    with engine_lock:
        engine = current_engine

    engine.start()

    update_count = 0
    total_audio_sec = 0.0
    total_infer_ms = 0.0
    prev_text = ""
    silence_streak = 0

    next_tick = time.monotonic() + interval
    while running:
        now = time.monotonic()
        sleep_for = max(0, next_tick - now)
        time.sleep(sleep_for)
        next_tick = time.monotonic() + interval
        if paused:
            drain_buffer()
            continue

        # Check for engine switch — serialized here so old model is freed
        # before new one loads (prevents both models in memory simultaneously)
        if pending_engine_name is not None:
            new_name = pending_engine_name
            pending_engine_name = None
            log(f"Engine switch: stopping {engine.name}, loading {new_name}...")
            engine.stop()
            del engine
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass
            new_engine = ENGINES[new_name]()
            try:
                new_engine.load()
            except Exception as e:
                log(f"Engine load failed: {e}, reverting...")
                broadcast({"type": "status", "content": f"Failed to load {new_name}"})
                new_engine = ENGINES["qwen"]()
                new_engine.load()
            engine = new_engine
            with engine_lock:
                current_engine = engine
            engine.start()
            prev_text = ""
            silence_streak = 0
            update_count = 0
            total_audio_sec = 0.0
            total_infer_ms = 0.0
            broadcast({"type": "engine", "engine": engine.name})
            broadcast({"type": "status", "content": f"Switched to {engine.name}"})
            continue

        chunk = drain_buffer()
        if chunk is None or len(chunk) < 800:
            silence_streak += 1
            if silence_streak >= SILENCE_STREAK_RESET and engine.needs_manual_reset():
                engine.reset()
                silence_streak = 0
            continue

        if not contains_speech(chunk, threshold=vad_threshold):
            silence_streak += 1
            if silence_streak >= SILENCE_STREAK_RESET and engine.needs_manual_reset():
                engine.reset()
                silence_streak = 0
            continue

        silence_streak = 0
        audio_sec = len(chunk) / SAMPLE_RATE

        t0 = time.perf_counter()
        try:
            stable, draft = engine.feed(chunk)
        except Exception as e:
            log(f"engine error: {e}")
            try:
                engine.reset()
            except Exception:
                pass
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000

        update_count += 1
        total_audio_sec += audio_sec
        total_infer_ms += elapsed_ms

        full_text = f"{stable} {draft}".strip()
        if not full_text or full_text == prev_text:
            continue
        prev_text = full_text

        broadcast({
            "type": "stream",
            "finalized": stable,
            "draft": draft,
        })

        avg_ms = total_infer_ms / update_count
        rtf = (total_infer_ms / 1000) / total_audio_sec if total_audio_sec > 0 else 0
        log(
            f"  [{elapsed_ms:.0f}ms | avg {avg_ms:.0f}ms | "
            f"RTF={rtf:.4f} | {engine.name}] "
            f"{stable[-50:]}|{draft[:35]}"
        )

    # Clean shutdown
    engine.stop()


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
.topbar .controls{display:flex;gap:0.5rem;align-items:center}
.btn{
  background:#1a1a1a;border:1px solid #2a2a2a;color:#888;
  padding:0.35rem 0.9rem;border-radius:4px;font-size:0.65rem;
  cursor:pointer;letter-spacing:0.06em;text-transform:uppercase;
  font-weight:500;transition:all 0.2s;
}
.btn:hover{background:#222;color:#ccc;border-color:#444}
.btn.active{background:#1a2e1a;border-color:#2a4a2a;color:#4ade80}
.btn.danger:hover{background:#2e1a1a;border-color:#4a2a2a;color:#f87171}
select.engine-select{
  background:#1a1a1a;border:1px solid #2a2a2a;color:#888;
  padding:0.35rem 0.5rem;border-radius:4px;font-size:0.65rem;
  cursor:pointer;letter-spacing:0.04em;
  font-weight:500;outline:none;
  -webkit-appearance:none;appearance:none;
}
select.engine-select:hover{background:#222;color:#ccc;border-color:#444}
select.engine-select:focus{border-color:#4ade80}
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
    <select class="engine-select" id="engineSelect" onchange="switchEngine(this.value)">
      <option value="qwen">Qwen3 0.6B</option>
      <option value="qwen-1.7b">Qwen3 1.7B</option>
    </select>
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
const engineSelect=document.getElementById('engineSelect');
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
function switchEngine(eng){
  statusEl.style.display='block';
  statusEl.textContent='Switching to '+eng+'...';
  textEl.innerHTML='';
  fetch('/switch?engine='+encodeURIComponent(eng));
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
  }else if(d.type==='engine'){
    engineSelect.value=d.engine;
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

// Set initial engine from server
fetch('/engine').then(r=>r.json()).then(d=>{engineSelect.value=d.engine});
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

        elif self.path.startswith("/switch"):
            global pending_engine_name
            from urllib.parse import urlparse, parse_qs
            params = parse_qs(urlparse(self.path).query)
            engine_name = params.get("engine", [""])[0]

            if engine_name not in ENGINES:
                self.send_response(400)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Unknown engine: {engine_name}".encode())
                return

            with engine_lock:
                if current_engine and current_engine.name == engine_name:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"already active")
                    return

            # Signal transcription loop to handle the switch (serialized:
            # old model freed before new one loads, preventing OOM)
            pending_engine_name = engine_name
            broadcast({"type": "status", "content": f"Switching to {engine_name}..."})
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"switching to {engine_name}".encode())
            log(f"Engine switch queued: {engine_name}")

        elif self.path == "/engine":
            with engine_lock:
                name = current_engine.name if current_engine else "unknown"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"engine": name}).encode())

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")  # Disable nginx/proxy buffering
            self.end_headers()
            self.wfile.write(b"retry: 3000\n\n")  # Auto-reconnect hint for EventSource
            # Send current status so new subscribers don't see stale "Loading model..."
            with engine_lock:
                eng_name = current_engine.name if current_engine else "unknown"
            welcome = f"data: {json.dumps({'type': 'status', 'content': 'Listening...'})}\n\n"
            self.wfile.write(welcome.encode())
            self.wfile.write(f"data: {json.dumps({'type': 'engine', 'engine': eng_name})}\n\n".encode())
            self.wfile.flush()

            q = queue.Queue(maxsize=50)
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
    global running, current_engine

    import platform

    if platform.machine() != "arm64" or platform.system() != "Darwin":
        print(
            "TextStream requires Apple Silicon (M1/M2/M3/M4). "
            "MLX does not run on Intel Macs or Linux.",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="TextStream - live speech to text")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--engine", default="qwen",
        choices=list(ENGINES.keys()),
        help="ASR engine (default: qwen)",
    )
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL,
        help=f"Seconds between streaming updates (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--vad-threshold", type=float, default=DEFAULT_VAD_THRESHOLD,
        help=f"Silero VAD speech probability threshold (default: {DEFAULT_VAD_THRESHOLD})",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-grafana", action="store_true", help="Disable Grafana push")
    args = parser.parse_args()

    if args.no_grafana or not GRAFANA_TOKEN:
        global push_annotation
        push_annotation = lambda text: None
    else:
        threading.Thread(target=_annotation_worker, daemon=True).start()

    # Limit MLX Metal cache to prevent memory pressure on 8/16GB Macs
    try:
        import mlx.core as mx
        mx.set_cache_limit(1024 * 1024 * 1024)  # 1GB
    except Exception:
        pass

    # Create and load initial engine
    current_engine = ENGINES[args.engine]()
    current_engine.load()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=0,  # Let CoreAudio choose optimal buffer size
        latency="low",  # Apple Silicon low-latency hint
    )
    stream.start()
    log("Microphone active")

    t = threading.Thread(
        target=transcription_loop, args=(args.interval, args.vad_threshold), daemon=True
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
        if not running:
            return
        running = False
        log("\nStopping...")

        def _cleanup():
            stream.stop()
            stream.close()
            close_transcript()
            httpd.shutdown()

        # Run cleanup in a thread to avoid deadlock when called from signal handler
        # (httpd.shutdown() waits for serve_forever() which runs on main thread)
        threading.Thread(target=_cleanup, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
