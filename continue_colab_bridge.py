"""Authenticated, non-streaming OpenAI-compatible bridge for a Colab llama-server."""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import requests


MAX_REQUEST_BYTES = 1_048_576
MAX_OUTPUT_TOKENS = 512
CONTINUE_OUTPUT_TOKENS = 128
CONTEXT_TOKENS = 4096
PROMPT_SAFETY_TOKENS = 64
BACKEND_TIMEOUT = (15, 110)
_TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False


class ColabContinueBridge:
    """Expose only authenticated /v1/models and non-streaming chat completions."""

    def __init__(
        self,
        base_url: str,
        model: str,
        gpu_lock: threading.Lock,
        is_blocked: Callable[[], bool],
        mark_blocked: Callable[[], None],
        api_key: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.gpu_lock = gpu_lock
        self.is_blocked = is_blocked
        self.mark_blocked = mark_blocked
        self.api_key = api_key or secrets.token_urlsafe(32)
        self.session = session or requests.Session()
        self.server: _Server | None = None
        self.server_thread: threading.Thread | None = None
        self.tunnel_process: subprocess.Popen[str] | None = None
        self.tunnel_reader: threading.Thread | None = None
        self.api_base: str | None = None
        self._tunnel_lock = threading.Lock()
        self.accepting = True

    def start_proxy(self) -> str:
        if not self.accepting:
            raise RuntimeError("Stopped bridge instances cannot be restarted; create a new connection.")
        if self.server is not None:
            return f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        server = _Server(("127.0.0.1", 0), self._make_handler())
        thread = threading.Thread(target=server.serve_forever, name="continue-proxy", daemon=True)
        thread.start()
        self.server = server
        self.server_thread = thread
        return f"http://127.0.0.1:{server.server_address[1]}/v1"

    def start_tunnel(self, cloudflared: str = "cloudflared", timeout: float = 45) -> str:
        with self._tunnel_lock:
            if self.api_base and self.tunnel_process and self.tunnel_process.poll() is None:
                return self.api_base
            if self.server is None:
                self.start_proxy()
            executable = shutil.which(cloudflared) if os.path.sep not in cloudflared else cloudflared
            if not executable or not Path(executable).is_file():
                raise RuntimeError("Cloudflare公式のcloudflaredが見つかりません。公式パッケージを準備して再試行してください。")
            local_url = f"http://127.0.0.1:{self.server.server_address[1]}"
            process = subprocess.Popen(
                [executable, "tunnel", "--no-autoupdate", "--url", local_url],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            url_event = threading.Event()
            result: dict[str, str] = {}

            def read_url() -> None:
                assert process.stdout is not None
                for line in process.stdout:
                    match = _TUNNEL_URL.search(line)
                    if match and not result:
                        result["url"] = match.group(0)
                        url_event.set()
                url_event.set()

            reader = threading.Thread(target=read_url, name="cloudflared-output", daemon=True)
            reader.start()
            if not url_event.wait(timeout) or "url" not in result:
                self._terminate_owned_process(process)
                reader.join(timeout=2)
                raise RuntimeError("Cloudflare Quick Tunnel URLを取得できませんでした。cloudflaredの終了状態を確認してください。")
            if process.poll() is not None:
                self._terminate_owned_process(process)
                reader.join(timeout=2)
                raise RuntimeError("Cloudflare Quick TunnelがURL取得後に終了しました。")
            self.tunnel_process = process
            self.tunnel_reader = reader
            self.api_base = result["url"] + "/v1"
            return self.api_base

    def write_config(self, path: str = "/content/colab_continue_connection.json") -> str:
        if not self.api_base:
            raise RuntimeError("先にVS Code接続を開始してください。")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "apiBase": self.api_base,
            "model": self.model,
            "apiKey": self.api_key,
            "continue": {
                "yaml": {"defaultCompletionOptions": {"contextLength": CONTEXT_TOKENS, "maxTokens": CONTINUE_OUTPUT_TOKENS, "stream": False}},
                "json": {"contextLength": CONTEXT_TOKENS, "completionOptions": {"maxTokens": CONTINUE_OUTPUT_TOKENS, "stream": False}},
            },
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            target.chmod(0o600)
        except OSError:
            pass
        return str(target)

    @staticmethod
    def _terminate_owned_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def stop_tunnel(self) -> None:
        with self._tunnel_lock:
            process = self.tunnel_process
            reader = self.tunnel_reader
            self.tunnel_process = None
            self.tunnel_reader = None
            self.api_base = None
            if process is not None:
                self._terminate_owned_process(process)
            if reader is not None:
                reader.join(timeout=2)

    def stop_proxy(self) -> None:
        server = self.server
        thread = self.server_thread
        self.server = None
        self.server_thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2)

    def stop(self) -> None:
        self.stop_accepting()
        self.session.close()

    def stop_accepting(self) -> None:
        self.accepting = False
        self.stop_tunnel()
        self.stop_proxy()

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(30)

            def log_message(self, _format: str, *args: object) -> None:
                return

            def _send(self, status: int, payload: bytes, content_type: str = "application/json") -> None:
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(payload)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                self.close_connection = True

            def _authorized(self) -> bool:
                if not bridge.accepting:
                    return False
                supplied = self.headers.get("Authorization", "")
                prefix = "Bearer "
                candidate = supplied[len(prefix):] if supplied.startswith(prefix) else ""
                return bool(candidate) and hmac.compare_digest(candidate, bridge.api_key)

            def _error(self, status: int, message: str) -> None:
                self._send(status, json.dumps({"error": {"message": message, "type": "invalid_request_error"}}).encode())

            def do_GET(self) -> None:
                if urllib.parse.urlsplit(self.path).path != "/v1/models":
                    self._error(404, "Not found")
                    return
                if not self._authorized():
                    self._error(401, "Unauthorized")
                    return
                try:
                    response = bridge.session.get(f"{bridge.base_url}/v1/models", timeout=(5, 15))
                    response.raise_for_status()
                    ids = [str(item.get("id")) for item in response.json().get("data", [])]
                    if bridge.model not in ids:
                        self._error(502, "Expected model alias is unavailable")
                        return
                    self._send(response.status_code, response.content)
                except requests.RequestException:
                    self._error(502, "Backend is unavailable")
                except (ValueError, TypeError, AttributeError):
                    self._error(502, "Backend returned an invalid model list")

            def do_POST(self) -> None:
                if urllib.parse.urlsplit(self.path).path != "/v1/chat/completions":
                    self._error(404, "Not found")
                    return
                if not self._authorized():
                    self._error(401, "Unauthorized")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    length = -1
                if length < 0 or length > MAX_REQUEST_BYTES:
                    self._error(413, "Request body is missing or too large")
                    return
                try:
                    body = json.loads(self.rfile.read(length))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self._error(400, "Invalid JSON request")
                    return
                if not isinstance(body, dict) or body.get("model") != bridge.model:
                    self._error(400, "Unexpected model alias")
                    return
                if body.get("stream", False) is not False:
                    self._error(400, "Streaming is disabled for this Quick Tunnel connection")
                    return
                messages = body.get("messages")
                if not isinstance(messages, list) or not messages:
                    self._error(400, "messages must be a non-empty array")
                    return
                try:
                    requested = body.get("max_tokens", body.get("max_completion_tokens", MAX_OUTPUT_TOKENS))
                    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1 or requested > MAX_OUTPUT_TOKENS:
                        self._error(400, "max_tokens must be between 1 and 512")
                        return
                    body["max_tokens"] = requested
                    body.pop("max_completion_tokens", None)
                    body["chat_template_kwargs"] = {"enable_thinking": False}
                    if not bridge.gpu_lock.acquire(blocking=False):
                        self._error(409, "GPU inference is busy")
                        return
                    try:
                        # Recheck only after taking the same lock used by notebook inference.
                        if not bridge.accepting:
                            self._error(401, "Unauthorized")
                            return
                        if bridge.is_blocked():
                            self._error(409, "Inference state is uncertain; stop and restart both backends")
                            return
                        if not self._check_context(messages, requested):
                            self._error(400, "Prompt and completion exceed the configured 4096-token context")
                            return
                        response = bridge.session.post(
                            f"{bridge.base_url}/v1/chat/completions",
                            json=body,
                            timeout=BACKEND_TIMEOUT,
                        )
                        response.raise_for_status()
                        payload = response.content
                        try:
                            decoded = json.loads(payload)
                            if decoded.get("model") not in (None, bridge.model):
                                raise ValueError("unexpected model")
                        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
                            bridge.mark_blocked()
                            self._error(502, "Backend returned an invalid completion; stop and restart both backends")
                            return
                        self._send(response.status_code, payload)
                    except requests.RequestException:
                        bridge.mark_blocked()
                        self._error(504, "Backend completion status is uncertain; stop and restart both backends")
                    finally:
                        bridge.gpu_lock.release()
                except (TypeError, ValueError):
                    self._error(400, "Invalid completion options")

            def _check_context(self, messages: list[object], max_tokens: int) -> bool:
                try:
                    template = bridge.session.post(
                        f"{bridge.base_url}/apply-template",
                        json={"messages": messages, "chat_template_kwargs": {"enable_thinking": False}},
                        timeout=(10, 30),
                    )
                    template.raise_for_status()
                    prompt = template.json().get("prompt")
                    if not isinstance(prompt, str):
                        return False
                    tokenized = bridge.session.post(
                        f"{bridge.base_url}/tokenize",
                        json={"content": prompt, "add_special": False},
                        timeout=(10, 30),
                    )
                    tokenized.raise_for_status()
                    tokens = tokenized.json().get("tokens")
                    return isinstance(tokens, list) and len(tokens) + max_tokens + PROMPT_SAFETY_TOKENS <= CONTEXT_TOKENS
                except (requests.RequestException, ValueError, TypeError, AttributeError):
                    return False

        return Handler
