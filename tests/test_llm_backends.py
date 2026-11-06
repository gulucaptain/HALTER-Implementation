"""Exercise the OpenAI-compatible client (API + local vLLM server modes) against a fake HTTP server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

openai = pytest.importorskip("openai")

from halter.llm.client import LLMConfig, make_llm  # noqa: E402

RECEIVED = []


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        RECEIVED.append(body)
        resp = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": '{"success": true}'}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        out = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def server():
    s = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_port}/v1"
    s.shutdown()


@pytest.mark.parametrize("backend", ["api", "local_server"])
def test_openai_compat(server, backend):
    RECEIVED.clear()
    llm = make_llm(LLMConfig(backend=backend, model="qwen-test", base_url=server, api_key_env=""))
    out = llm.chat([{"role": "user", "content": "hi", "_ctx": {"secret": 1}}])
    assert json.loads(out)["success"] is True
    body = RECEIVED[0]
    assert "_ctx" not in body["messages"][0]
    if backend == "local_server":
        assert body["chat_template_kwargs"]["enable_thinking"] is False
    else:
        assert body["enable_thinking"] is False
