import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

import requests

from continue_colab_bridge import ColabContinueBridge


MODEL = "ternary-bonsai2-27b-abliterated-pq2-gguf"


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.chat_entered = threading.Event()
        self.chat_release = threading.Event()
        self.block_chat = False
        self.chat_body = None
        self.template_body = None
        self.response_content = "ok"
        self.response_payload = None

        owner = self

        class Backend(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def send_json(self, status, value):
                payload = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path == "/v1/models":
                    self.send_json(200, {"data": [{"id": MODEL}]})
                else:
                    self.send_json(404, {})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/apply-template":
                    owner.template_body = body
                    self.send_json(200, {"prompt": "mock prompt"})
                elif self.path == "/tokenize":
                    self.send_json(200, {"tokens": [1] * owner.prompt_tokens})
                elif self.path == "/v1/chat/completions":
                    owner.chat_body = body
                    owner.chat_entered.set()
                    if owner.block_chat:
                        owner.chat_release.wait(2)
                    payload = owner.response_payload or {
                        "model": MODEL,
                        "choices": [{"message": {"content": owner.response_content}}],
                    }
                    self.send_json(200, payload)
                else:
                    self.send_json(404, {})

        self.prompt_tokens = 10
        self.backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        self.backend.daemon_threads = True
        self.backend_thread = threading.Thread(target=self.backend.serve_forever, daemon=True)
        self.backend_thread.start()
        self.gpu_lock = threading.Lock()
        self.blocked = False
        self.bridge = ColabContinueBridge(
            f"http://127.0.0.1:{self.backend.server_address[1]}",
            MODEL,
            self.gpu_lock,
            lambda: self.blocked,
            self._mark_blocked,
            api_key="test-secret",
        )
        self.base = self.bridge.start_proxy()
        self.client = requests.Session()

    def tearDown(self):
        self.chat_release.set()
        self.bridge.stop()
        self.client.close()
        self.backend.shutdown()
        self.backend.server_close()
        self.backend_thread.join(timeout=2)

    def _mark_blocked(self):
        self.blocked = True

    def headers(self, key="test-secret"):
        return {"Authorization": f"Bearer {key}"}

    def body(self, **changes):
        value = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False}
        value.update(changes)
        return value

    def test_auth_routes_alias_and_nonstream_config(self):
        no_auth = self.client.get(self.base + "/models")
        self.assertEqual(no_auth.status_code, 401)
        self.assertNotIn("test-secret", no_auth.text)
        models = self.client.get(self.base + "/models", headers=self.headers())
        self.assertEqual(models.status_code, 200)
        self.assertEqual(models.json()["data"][0]["id"], MODEL)
        self.assertEqual(self.client.get(self.base + "/other", headers=self.headers()).status_code, 404)
        bad_model = self.client.post(self.base + "/chat/completions", headers=self.headers(), json=self.body(model="animagine-xl-3.1"))
        self.assertEqual(bad_model.status_code, 400)
        streaming = self.client.post(self.base + "/chat/completions", headers=self.headers(), json=self.body(stream=True))
        self.assertEqual(streaming.status_code, 400)
        too_many = self.client.post(self.base + "/chat/completions", headers=self.headers(), json=self.body(max_tokens=1025))
        self.assertEqual(too_many.status_code, 400)
        result = self.client.post(
            self.base + "/chat/completions",
            headers=self.headers(),
            json=self.body(max_tokens=128, reasoning_effort="none", thinking_budget_tokens=-1,
                           chat_template_kwargs={"enable_thinking": False}),
        )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["choices"][0]["message"]["content"], "ok")
        self.assertEqual(self.chat_body["reasoning_effort"], "medium")
        self.assertEqual(self.chat_body["thinking_budget_tokens"], 128)
        self.assertEqual(self.chat_body["chat_template_kwargs"], {"reasoning_effort": "medium"})
        self.assertEqual(self.template_body["chat_template_kwargs"], {"reasoning_effort": "medium"})
        self.assertEqual(self.chat_body["max_tokens"], 128)
        default_result = self.client.post(self.base + "/chat/completions", headers=self.headers(), json=self.body())
        self.assertEqual(default_result.status_code, 200)
        self.assertEqual(self.chat_body["max_tokens"], 768)
        with TemporaryDirectory() as directory:
            self.bridge.api_base = "https://example.trycloudflare.com/v1"
            path = self.bridge.write_config(str(Path(directory) / "connection.json"))
            config = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(config["continue"]["yaml"]["defaultCompletionOptions"], {"contextLength": 4096, "maxTokens": 768, "stream": False})
            self.assertEqual(config["continue"]["json"]["completionOptions"], {"maxTokens": 768, "stream": False})
            self.assertEqual(config["apiKey"], "test-secret")

    def test_bridge_rejects_empty_or_reasoning_only_visible_content(self):
        self.response_content = ""
        empty = self.client.post(self.base + "/chat/completions", headers=self.headers(), json=self.body())
        self.assertEqual(empty.status_code, 502)
        self.response_content = "unfinished thought </think>"
        thought_only = self.client.post(self.base + "/chat/completions", headers=self.headers(), json=self.body())
        self.assertEqual(thought_only.status_code, 502)

    def test_bridge_rejects_malformed_or_missing_visible_content(self):
        for payload in (
            {"model": MODEL, "choices": [None]},
            {"model": MODEL, "choices": [{"message": {"content": None}}]},
        ):
            with self.subTest(payload=payload):
                self.response_payload = payload
                result = self.client.post(self.base + "/chat/completions", headers=self.headers(), json=self.body())
                self.assertEqual(result.status_code, 502)

    def test_lock_rejects_overlap_and_context_budget(self):
        self.block_chat = True
        first = {}

        def call_first():
            first["response"] = self.client.post(self.base + "/chat/completions", headers=self.headers(), json=self.body())

        worker = threading.Thread(target=call_first)
        worker.start()
        self.assertTrue(self.chat_entered.wait(1))
        self.assertTrue(self.gpu_lock.locked())
        overlap = self.client.post(self.base + "/chat/completions", headers=self.headers(), json=self.body())
        self.assertEqual(overlap.status_code, 409)
        self.chat_release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(first["response"].status_code, 200)
        self.prompt_tokens = 4000
        oversized = self.client.post(self.base + "/chat/completions", headers=self.headers(), json=self.body(max_tokens=64))
        self.assertEqual(oversized.status_code, 400)
        self.assertFalse(self.gpu_lock.locked())

    def test_backend_timeout_blocks_future_chat(self):
        original_post = self.bridge.session.post

        def timeout_chat(url, *args, **kwargs):
            if url.endswith("/v1/chat/completions"):
                raise requests.ReadTimeout("simulated backend timeout")
            return original_post(url, *args, **kwargs)

        self.bridge.session.post = timeout_chat
        timed_out = self.client.post(self.base + "/chat/completions", headers=self.headers(), json=self.body())
        self.assertEqual(timed_out.status_code, 504)
        self.assertTrue(self.blocked)
        self.assertFalse(self.gpu_lock.locked())
        blocked = self.client.post(self.base + "/chat/completions", headers=self.headers(), json=self.body())
        self.assertEqual(blocked.status_code, 409)

    def test_stop_closes_listener_and_prevents_old_bridge_restart(self):
        self.bridge.stop_accepting()
        self.assertFalse(self.bridge.accepting)
        with self.assertRaisesRegex(RuntimeError, "cannot be restarted"):
            self.bridge.start_proxy()


if __name__ == "__main__":
    unittest.main()
