"""HTTP server tối giản cho TTS (chỉ stdlib — không thêm dependency).

Endpoints:
- `GET /health` → `{"status": "ok", "version": ...}`
- `GET /info`   → sample rate, dialect mặc định, max_chars...
- `POST /synthesize` JSON `{"text": ..., "dialect"?, "speed"?, "overrides"?,
  "peak_normalize"?, "max_chars"?}` → bytes WAV (`audio/wav`).

Server chỉ phù hợp nội bộ/nghiên cứu: một model đã nạp, một lock tuần tự,
không auth. Production nên thêm reverse proxy + hàng đợi.
"""

from __future__ import annotations

import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from vifasttts import __version__
from vifasttts.infer import Synthesizer
from vifasttts.runtime.postprocess import postprocess

MAX_BODY_BYTES = 64 * 1024
MAX_TEXT_CHARS = 2000


def wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Waveform numpy float32 → bytes WAV PCM 16-bit mono."""
    pcm = (np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0) * 32767.0).round().astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())
    return buffer.getvalue()


class TtsEngine:
    """Nạp model một lần; synthesize an toàn đa luồng bằng lock."""

    def __init__(
        self,
        config: dict,
        checkpoint: str,
        vocoder_checkpoint: str | None = None,
        *,
        device: str | None = None,
    ) -> None:
        self.synthesizer = Synthesizer(config, checkpoint, vocoder_checkpoint, device=device)
        self.lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self.synthesizer.sample_rate

    def synthesize_bytes(self, payload: dict[str, Any]) -> bytes:
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("Thiếu trường 'text'")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"Văn bản quá dài (> {MAX_TEXT_CHARS} ký tự)")
        dialect = str(payload.get("dialect", "north"))
        speed = float(payload.get("speed", 1.0))
        if not 0.25 <= speed <= 4.0:
            raise ValueError("speed phải trong khoảng 0.25–4.0")
        overrides = payload.get("overrides")
        if overrides is not None and not isinstance(overrides, dict):
            raise ValueError("'overrides' phải là object chuỗi→chuỗi")
        max_chars = int(payload.get("max_chars", 400))
        peak_normalize = bool(payload.get("peak_normalize", False))

        with self.lock:
            pieces = [audio for audio, _ in self.synthesizer.iter_synthesize(
                text, dialect=dialect, speed=speed,
                overrides={str(k): str(v) for k, v in overrides.items()} if overrides else None,
                max_chars=max_chars,
            )]
        waveform = postprocess(
            np.concatenate(pieces), self.sample_rate,
            target_db=-3.0 if peak_normalize else None,
        )
        return wav_bytes(waveform, self.sample_rate)


def make_handler(engine: TtsEngine):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"ViFastTts/{__version__}"

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "version": __version__})
            elif self.path == "/info":
                self._send_json(200, {
                    "version": __version__,
                    "sample_rate": engine.sample_rate,
                    "max_text_chars": MAX_TEXT_CHARS,
                    "default_dialect": "north",
                })
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/synthesize":
                self._send_json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_json(400, {"error": "body rỗng hoặc quá lớn"})
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise TypeError("body phải là JSON object")
                audio = engine.synthesize_bytes(payload)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001 - server phải trả 500 thay vì sập
                self._send_json(500, {"error": f"lỗi tổng hợp: {exc}"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio)))
            self.end_headers()
            self.wfile.write(audio)

        def log_message(self, *args: Any) -> None:  # im lặng trong test/log
            pass

    return Handler


class _TtsHttpServer(ThreadingHTTPServer):
    """ThreadingHTTPServer bỏ reverse-DNS trong `server_bind`.

    `HTTPServer.server_bind` gọi `socket.getfqdn()` — có thể treo hàng chục giây
    khi mạng chặn DNS ngược; ở đây chỉ bind và lấy cổng thật.
    """

    def server_bind(self) -> None:
        import socketserver

        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


def create_server(engine: TtsEngine, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return _TtsHttpServer((host, port), make_handler(engine))


def serve(
    config: dict,
    checkpoint: str,
    *,
    vocoder_checkpoint: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str | None = None,
) -> None:
    """Chạy server tới khi Ctrl-C (blocking)."""
    engine = TtsEngine(config, checkpoint, vocoder_checkpoint, device=device)
    httpd = create_server(engine, host, port)
    print(f"ViFastTts server: http://{host}:{httpd.server_address[1]} (Ctrl-C để dừng)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - tương tác người dùng
        pass
    finally:
        httpd.server_close()
