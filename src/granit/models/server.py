"""Start, health-check and stop ``mlx_lm.server`` (Phase B, PLAN.md §2.1).

The server runs as its own process so stopping it returns all of its memory. M1 uses this to time phase
switches; M5 builds the phase manager on top of it.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from granit.config import (
    LLM_HOST,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MLX_CACHE_LIMIT_GB,
    LLM_PORT,
    LLM_PROMPT_CACHE_BYTES,
)


@dataclass(frozen=True)
class ServerConfig:
    model_path: Path
    host: str = LLM_HOST
    port: int = LLM_PORT
    max_tokens: int = LLM_MAX_OUTPUT_TOKENS
    # Memory limits measured in M1 (PLAN.md §3.3). None means mlx-lm's own default (unbounded).
    prompt_cache_bytes: str | None = LLM_PROMPT_CACHE_BYTES
    mlx_cache_limit_gb: float | None = LLM_MLX_CACHE_LIMIT_GB
    extra_args: tuple[str, ...] = field(default=())

    def command(self) -> list[str]:
        if self.mlx_cache_limit_gb is None:
            cmd = [sys.executable, "-m", "mlx_lm", "server"]
        else:
            cmd = [sys.executable, "-m", "granit.models.mlx_server"]
            cmd += ["--mlx-cache-limit-gb", str(self.mlx_cache_limit_gb)]
        cmd += ["--model", str(self.model_path), "--host", self.host, "--port", str(self.port)]
        cmd += ["--max-tokens", str(self.max_tokens)]
        if self.prompt_cache_bytes:
            cmd += ["--prompt-cache-bytes", self.prompt_cache_bytes]
        return cmd + list(self.extra_args)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class ServerError(RuntimeError):
    pass


class LLMServer:
    """One ``mlx_lm.server`` process. Use as a context manager so it's always stopped."""

    def __init__(self, config: ServerConfig, log_path: Path | None = None) -> None:
        self.config = config
        self.log_path = log_path
        self.process: subprocess.Popen[bytes] | None = None
        self._log: IO[bytes] | None = None

    @property
    def pid(self) -> int:
        if self.process is None:
            raise ServerError("server is not running")
        return self.process.pid

    def start(self) -> None:
        if self.process is not None:
            raise ServerError("server already started")
        if self.log_path:
            self._log = self.log_path.open("ab")
        self.process = subprocess.Popen(
            self.config.command(),
            stdout=self._log or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.config.base_url}/health", timeout=2) as response:
                return response.status == 200
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            return False

    def wait_ready(self, timeout: float = 120, poll: float = 0.1) -> float:
        """Block until ``/health`` answers 200; returns seconds waited. Raises if the process dies."""
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            if self.process is not None and self.process.poll() is not None:
                raise ServerError(f"mlx_lm.server exited with code {self.process.returncode}")
            if self.healthy():
                return time.perf_counter() - start
            time.sleep(poll)
        raise ServerError(f"mlx_lm.server not ready after {timeout:.0f} s")

    def stop(self, timeout: float = 30) -> float:
        """SIGTERM, then SIGKILL after ``timeout``; returns seconds until the process exited."""
        if self.process is None:
            return 0.0
        start = time.perf_counter()
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
        if self._log:
            self._log.close()
            self._log = None
        return time.perf_counter() - start

    def __enter__(self) -> LLMServer:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    ttft_s: float  # time to first streamed token (≈ prefill time)
    total_s: float

    @property
    def decode_tps(self) -> float:
        decode_time = self.total_s - self.ttft_s
        return (self.completion_tokens - 1) / decode_time if decode_time > 0 else 0.0

    @property
    def prefill_tps(self) -> float:
        uncached = self.prompt_tokens - self.cached_tokens
        return uncached / self.ttft_s if self.ttft_s > 0 else 0.0


def stream_chat(
    base_url: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 256,
    enable_thinking: bool | None = None,
    timeout: float = 600,
) -> Completion:
    """POST a streaming chat completion (OpenAI API) and time it. Counts reasoning and content tokens alike.

    ``enable_thinking=None`` keeps the model's default (Granite 4.2 thinks unless told not to).
    """
    body: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first: float | None = None
    parts: list[str] = []
    usage: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for event in _sse_events(response):
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning") or "")
                if piece:
                    first = first or time.perf_counter()
                    parts.append(piece)
    end = time.perf_counter()
    details = usage.get("prompt_tokens_details") or {}
    return Completion(
        text="".join(parts),
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        cached_tokens=int(details.get("cached_tokens", 0)),
        ttft_s=(first or end) - start,
        total_s=end - start,
    )


def _sse_events(response: Any) -> Iterator[dict[str, Any]]:
    for raw in response:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return
        yield json.loads(data)
