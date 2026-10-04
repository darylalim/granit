"""mlx_lm.server manager and streaming client, against a fake OpenAI-compatible server (no model)."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from granit import config
from granit.models import mlx_server
from granit.models.server import Completion, LLMServer, ServerConfig, ServerError, stream_chat


class FakeOpenAI(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []  # noqa: RUF012 (shared on purpose: the test reads it)

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenAI.requests.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [
            {"choices": [{"delta": {"reasoning": "hmm"}}]},
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 40},
                },
            },
        ]
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.01)
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def fake_server() -> Iterator[str]:
    FakeOpenAI.requests = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_stream_chat_collects_text_usage_and_timing(fake_server: str) -> None:
    completion = stream_chat(fake_server, [{"role": "user", "content": "hi"}], max_tokens=8)
    assert completion.text == "hmmHello world"
    assert (completion.prompt_tokens, completion.completion_tokens) == (100, 3)
    assert completion.cached_tokens == 40
    assert 0 < completion.ttft_s <= completion.total_s
    body = FakeOpenAI.requests[0]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["temperature"] == 0.0
    assert "chat_template_kwargs" not in body  # model default unless asked


def test_stream_chat_can_turn_thinking_off(fake_server: str) -> None:
    stream_chat(fake_server, [{"role": "user", "content": "hi"}], enable_thinking=False)
    assert FakeOpenAI.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_completion_rates() -> None:
    c = Completion(
        "x", prompt_tokens=1100, completion_tokens=101, cached_tokens=100, ttft_s=2.0, total_s=4.0
    )
    assert c.prefill_tps == 500.0  # only uncached tokens are prefilled
    assert c.decode_tps == 50.0  # 100 tokens after the first, over 2 s
    instant = Completion("", 0, 1, 0, ttft_s=0.0, total_s=0.0)
    assert (instant.prefill_tps, instant.decode_tps) == (0.0, 0.0)


def test_server_command_is_localhost_and_pinned_model(tmp_path: Path) -> None:
    cmd = ServerConfig(tmp_path / "model", port=9001).command()
    assert cmd[cmd.index("--model") + 1] == str(tmp_path / "model")
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "9001"
    assert ServerConfig(tmp_path, port=9001).base_url == "http://127.0.0.1:9001"


def test_server_uses_granits_memory_limits_by_default(tmp_path: Path) -> None:
    cmd = ServerConfig(tmp_path).command()
    assert cmd[:3] == [sys.executable, "-m", "granit.models.mlx_server"]
    assert cmd[cmd.index("--mlx-cache-limit-gb") + 1] == str(config.LLM_MLX_CACHE_LIMIT_GB)
    assert cmd[cmd.index("--prompt-cache-bytes") + 1] == config.LLM_PROMPT_CACHE_BYTES
    assert cmd[cmd.index("--max-tokens") + 1] == str(config.LLM_MAX_OUTPUT_TOKENS)
    assert (
        config.LLM_PROMPT_BUDGET_TOKENS + config.LLM_MAX_OUTPUT_TOKENS == config.LLM_CONTEXT_TOKENS
    )


def test_server_can_run_with_mlx_lm_defaults(tmp_path: Path) -> None:
    cmd = ServerConfig(tmp_path, prompt_cache_bytes=None, mlx_cache_limit_gb=None).command()
    assert cmd[:4] == [sys.executable, "-m", "mlx_lm", "server"]
    assert "--prompt-cache-bytes" not in cmd
    assert "--mlx-cache-limit-gb" not in cmd


def test_launcher_splits_its_own_flag_from_server_args() -> None:
    limit, rest = mlx_server.split_args(
        ["--mlx-cache-limit-gb", "1.5", "--model", "m", "--port", "1"]
    )
    assert limit == 1.5
    assert rest == ["--model", "m", "--port", "1"]
    assert mlx_server.split_args(["--model", "m"]) == (None, ["--model", "m"])


def test_server_lifecycle_with_a_stand_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_server: str
) -> None:
    """start → wait_ready (health 200) → stop, using a sleeping process instead of mlx_lm."""
    port = int(fake_server.rsplit(":", 1)[1])
    config = ServerConfig(tmp_path, port=port)
    monkeypatch.setattr(
        ServerConfig, "command", lambda _self: [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    server = LLMServer(config, log_path=tmp_path / "server.log")
    with server:
        assert server.wait_ready(timeout=5) < 5
        pid = server.pid
        assert pid > 0
    assert server.process is None
    with pytest.raises(ServerError):
        _ = server.pid


def test_wait_ready_reports_a_crashed_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ServerConfig, "command", lambda _self: [sys.executable, "-c", "raise SystemExit(3)"]
    )
    server = LLMServer(ServerConfig(tmp_path, port=1))
    server.start()
    with pytest.raises(ServerError, match="exited with code 3"):
        server.wait_ready(timeout=10)
    server.stop()


def test_stop_kills_a_process_that_ignores_sigterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubborn = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('up', flush=True); time.sleep(30)"
    monkeypatch.setattr(ServerConfig, "command", lambda _self: [sys.executable, "-c", stubborn])
    server = LLMServer(ServerConfig(tmp_path, port=1), log_path=tmp_path / "log")
    server.start()
    deadline = time.time() + 10
    while "up" not in (tmp_path / "log").read_text() and time.time() < deadline:
        time.sleep(0.05)
    assert server.stop(timeout=0.5) < 5
