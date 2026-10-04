"""``granit bench`` orchestrator: runs each phase in its own process, one phase at a time (PLAN.md §2.1).

Scenarios
- ``phase-a``: ingest worker load + workload (audio, a page, Vision, chunk embedding)
- ``phase-b``: mlx_lm.server (granit's memory limits) startup, decode tok/s, prefill / TTFT by context size, plus the UI backend's
  embedder + reranker + vector matrix running alongside it (their peaks add up to Phase B)
- ``prompt-cache``: server memory over 10 distinct RAG prompts, mlx-lm defaults vs granit's limits
- ``long-context``: one request filling the context cap on a fresh server, mlx-lm defaults vs granit's limits
- ``phase-c``: Guardian q8 no-think time per check
- ``switches``: B→A and A→B switch time, as the phase manager will do it
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import IO, Any

from granit.bench.stats import budget
from granit.config import HUB_MODELS, LLM_CONTEXT_TOKENS, PROJECT_ROOT
from granit.fixtures import synthetic_text
from granit.models.download import local_snapshot
from granit.models.memory import footprint
from granit.models.server import LLMServer, ServerConfig, stream_chat

Details = dict[str, Any]
RESULTS_DIR = PROJECT_ROOT / "bench" / "results"
CHARS_PER_TOKEN = (
    3.75  # measured on synthetic_text with the Granite tokenizer (8,488 tokens / 32,000 chars)
)
# The last size fills the context cap (leaving room for the 128 output tokens the bench asks for): the worst case.
AT_LIMIT_TOKENS = LLM_CONTEXT_TOKENS - 512  # margin: the chars-per-token estimate runs ~0.6% long
CONTEXT_TOKENS = (1_000, 4_000, 8_000, AT_LIMIT_TOKENS)
QUICK_CONTEXT_TOKENS = (1_000, 4_000)
RAG_PROMPT_TOKENS = 3_000  # 8 chunks × ~300 tokens + question + instructions


@dataclass
class Bench:
    quick: bool = False
    log_dir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="granit-bench-")))
    port: int = 8799  # not the app's port, so a bench never collides with a running app

    def server(self, name: str, mlx_lm_defaults: bool = False) -> LLMServer:
        """granit's server settings, or mlx-lm's unbounded defaults (no prompt-cache cap, no MLX cache limit)."""
        model = local_snapshot(HUB_MODELS["llm"])
        if mlx_lm_defaults:
            config = ServerConfig(
                model, port=self.port, prompt_cache_bytes=None, mlx_cache_limit_gb=None
            )
        else:
            config = ServerConfig(model, port=self.port)
        return LLMServer(config, log_path=self.log_dir / f"server-{name}.log")


# ── worker processes ──


class WorkerProcess:
    """A ``granit.bench.workers`` child; ``hold=True`` uses the READY / exit handshake."""

    def __init__(self, bench: Bench, name: str, hold: bool) -> None:
        cmd = [sys.executable, "-m", "granit.bench.workers", name]
        cmd += ["--hold"] if hold else []
        cmd += ["--quick"] if bench.quick else []
        self.stderr: IO[str] = (bench.log_dir / f"worker-{name}.log").open("a")
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            env=env,
        )
        self.started = time.perf_counter()

    def wait_ready(self) -> float:
        """Seconds from spawn until the worker printed READY (models loaded)."""
        assert self.proc.stdout
        for line in self.proc.stdout:
            if line.strip() == "READY":
                return time.perf_counter() - self.started
        raise RuntimeError(f"worker exited before READY (code {self.proc.wait()})")

    def finish(self, release: bool = False) -> tuple[float, Details]:
        """Tell a held worker to exit (if ``release``); returns (seconds until exit, its RESULT)."""
        assert self.proc.stdin and self.proc.stdout
        start = time.perf_counter()
        if release:
            self.proc.stdin.write("exit\n")
            self.proc.stdin.flush()
        result: Details = {}
        for line in self.proc.stdout:
            if line.startswith("RESULT "):
                result = json.loads(line[len("RESULT ") :])
        code = self.proc.wait()
        seconds = time.perf_counter() - start
        self.stderr.close()
        if code != 0 or not result:
            raise RuntimeError(f"worker failed with exit code {code}; see {self.stderr.name}")
        return seconds, result


def run_worker(bench: Bench, name: str) -> Details:
    return WorkerProcess(bench, name, hold=False).finish()[1]


# ── LLM helpers ──


def rag_messages(tokens: int, seed: int) -> list[dict[str, Any]]:
    """A document of about ``tokens`` tokens plus an instruction; each seed gives distinct text (no cache hit)."""
    document = synthetic_text(int(tokens * CHARS_PER_TOKEN), seed=seed)
    return [
        {
            "role": "user",
            "content": f"{document}\n\nList the action items above as five short bullet points.",
        }
    ]


def warm_up(server: LLMServer) -> float:
    """mlx_lm.server loads weights lazily on the first request; 'ready' needs one tiny request."""
    start = time.perf_counter()
    stream_chat(
        server.config.base_url,
        [{"role": "user", "content": "Hi"}],
        max_tokens=1,
        enable_thinking=False,
    )
    return time.perf_counter() - start


# ── scenarios ──


def phase_a(bench: Bench) -> Details:
    return run_worker(bench, "phase-a")


def phase_b(bench: Bench) -> Details:
    with bench.server("phase-b") as server:
        ready_s = server.wait_ready()
        warm_s = warm_up(server)
        # The UI backend's embedder + reranker: benchmarked while the server is idle, then held with a
        # 150K-vector matrix on MPS during the LLM tests (the realistic Phase B memory state).
        retrieval = WorkerProcess(bench, "retrieval", hold=True)
        retrieval.wait_ready()

        # Decode speed at a short context, thinking off so the output length is fixed by max_tokens.
        decode = [
            stream_chat(
                server.config.base_url,
                [{"role": "user", "content": f"Write a long, detailed story about granite #{i}."}],
                max_tokens=256,
                enable_thinking=False,
            )
            for i in range(3)
        ]
        contexts = []
        sizes = QUICK_CONTEXT_TOKENS if bench.quick else CONTEXT_TOKENS
        for i, tokens in enumerate(sizes):
            c = stream_chat(
                server.config.base_url,
                rag_messages(tokens, seed=1000 + i),
                max_tokens=128,
                enable_thinking=False,
            )
            fp = footprint(server.pid)
            contexts.append(
                {
                    "prompt_tokens": c.prompt_tokens,
                    "cached_tokens": c.cached_tokens,
                    "ttft_s": round(c.ttft_s, 2),
                    "prefill_tps": round(c.prefill_tps, 1),
                    "decode_tps": round(c.decode_tps, 1),
                    "server_footprint_gb": round(fp.current_gb, 2),
                }
            )
        server_peak = footprint(server.pid).peak_gb
        stop_s = server.stop()
    # Released only after the server stopped: the worker then runs its 1M-vector what-if alone.
    _, retrieval_result = retrieval.finish(release=True)

    retrieval_peak = retrieval_result["memory"]["peak_with_150k_matrix_gb"]
    return {
        "server": {
            "ready_s": round(ready_s, 2),
            "warm_up_s": round(warm_s, 2),
            "stop_s": round(stop_s, 2),
            "peak_footprint_gb": round(server_peak, 2),
        },
        "decode_tps_short_context": sorted(round(c.decode_tps, 1) for c in decode)[1],
        "contexts": contexts,
        "retrieval": retrieval_result,
        "peak_footprint_gb": round(server_peak + retrieval_peak, 2),
        "peak_note": "server peak + UI-backend retrieval peak (with a 150K-chunk matrix); an upper bound",
    }


def prompt_cache(bench: Bench) -> Details:
    """mlx_lm.server keeps up to 10 prompt KV caches with no byte limit by default (160 KB per token)."""
    n = 4 if bench.quick else 10
    out: Details = {}
    for label, mlx_lm_defaults in (("mlx_lm_defaults", True), ("granit", False)):
        with bench.server(f"cache-{label}", mlx_lm_defaults) as server:
            server.wait_ready()
            warm_up(server)
            after = []
            for i in range(n):
                stream_chat(
                    server.config.base_url,
                    rag_messages(RAG_PROMPT_TOKENS, seed=2000 + i),
                    max_tokens=64,
                    enable_thinking=False,
                )
                after.append(round(footprint(server.pid).current_gb, 2))
            out[label] = {
                "prompt_cache_bytes": server.config.prompt_cache_bytes,
                "mlx_cache_limit_gb": server.config.mlx_cache_limit_gb,
                "footprint_after_each_gb": after,
                "peak_footprint_gb": round(footprint(server.pid).peak_gb, 2),
            }
    return out


def long_context(bench: Bench) -> Details:
    """One request filling the context cap on a fresh server: the single-request peak, mlx-lm defaults vs
    granit's limits. (M1 also showed --prefill-step-size 512 vs 2048 makes no difference to this peak.)
    """
    out: Details = {}
    tokens = 8_000 if bench.quick else AT_LIMIT_TOKENS
    for label, mlx_lm_defaults in (("mlx_lm_defaults", True), ("granit", False)):
        with bench.server(f"long-{label}", mlx_lm_defaults) as server:
            server.wait_ready()
            warm_up(server)
            c = stream_chat(
                server.config.base_url,
                rag_messages(tokens, seed=3000),
                max_tokens=128,
                enable_thinking=False,
            )
            out[label] = {
                "prompt_tokens": c.prompt_tokens,
                "ttft_s": round(c.ttft_s, 2),
                "prefill_tps": round(c.prefill_tps, 1),
                "decode_tps": round(c.decode_tps, 1),
                "peak_footprint_gb": round(footprint(server.pid).peak_gb, 2),
            }
    return out


def phase_c(bench: Bench) -> Details:
    return run_worker(bench, "guardian")


def switches(bench: Bench) -> Details:
    """Time switches the way the phase manager will do them (§2.1), with weights in the OS file cache."""
    cycles = []
    server = bench.server("switches")
    server.start()
    server.wait_ready()
    warm_up(server)
    for _ in range(1 if bench.quick else 2):
        # B → A: stop the server (frees its memory), start the ingest worker, wait until its models are loaded.
        stop_s = server.stop()
        worker = WorkerProcess(bench, "phase-a", hold=True)
        load_a_s = worker.wait_ready()
        # A → B: the worker exits after its batch, then the server starts and answers a first request.
        exit_a_s, _ = worker.finish(release=True)
        server = bench.server("switches")
        server.start()
        ready_s = server.wait_ready()
        warm_s = warm_up(server)
        cycles.append(
            {
                "b_to_a_s": round(stop_s + load_a_s, 2),
                "a_to_b_s": round(exit_a_s + ready_s + warm_s, 2),
                "parts": {
                    "server_stop_s": round(stop_s, 2),
                    "phase_a_load_s": round(load_a_s, 2),
                    "phase_a_exit_s": round(exit_a_s, 2),
                    "server_ready_s": round(ready_s, 2),
                    "server_warm_up_s": round(warm_s, 2),
                },
            }
        )
    server.stop()
    return {"cycles": cycles}


SCENARIOS: dict[str, Callable[[Bench], Details]] = {
    "phase-a": phase_a,
    "phase-b": phase_b,
    "prompt-cache": prompt_cache,
    "long-context": long_context,
    "phase-c": phase_c,
    "switches": switches,
}


# ── report ──


def system_info() -> Details:
    from granit.models.memory import gpu_working_set_limit_gb

    def sysctl(name: str) -> str:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
    ).stdout.strip()
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": sha,
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_gb": round(int(sysctl("hw.memsize") or 0) / 2**30),
        "macos": platform.mac_ver()[0],
        "gpu_working_set_limit_gb": round(gpu_working_set_limit_gb(), 2),
        "versions": {
            p: version(p) for p in ("mlx", "mlx-lm", "mlx-vlm", "mlx-audio", "docling", "torch")
        },
        "models": {key: m.revision[:10] for key, m in HUB_MODELS.items()},
    }


def phase_peaks(results: Details) -> dict[str, float]:
    peaks = {}
    if "phase-a" in results:
        peaks["A"] = results["phase-a"]["memory"]["peak_footprint_gb"]
    if "phase-b" in results:
        peaks["B"] = results["phase-b"]["peak_footprint_gb"]
    if "phase-c" in results:
        peaks["C"] = results["phase-c"]["memory"]["peak_footprint_gb"]
    return peaks


def run(names: list[str], quick: bool, out_dir: Path = RESULTS_DIR) -> tuple[Details, Path]:
    bench = Bench(quick=quick)
    print(f"logs: {bench.log_dir}", flush=True)
    report: Details = {"system": system_info(), "quick": quick, "results": {}, "seconds": {}}
    for name in names:
        print(f"→ {name} …", flush=True)
        start = time.perf_counter()
        report["results"][name] = SCENARIOS[name](bench)
        report["seconds"][name] = round(time.perf_counter() - start, 1)
        print(f"  done in {report['seconds'][name]} s", flush=True)
    limit = report["system"]["gpu_working_set_limit_gb"]
    checks = budget(phase_peaks(report["results"]), limit)
    report["budget"] = [{"line": c.line(), "within_limit": c.within_limit} for c in checks]

    out_dir.mkdir(parents=True, exist_ok=True)
    # A subset run gets its own file, so it never overwrites a full run from the same commit.
    suffix = "" if names == list(SCENARIOS) else "-" + "+".join(names)
    suffix += "-quick" if quick else ""
    path = out_dir / f"{report['system']['date'][:10]}-{report['system']['git_sha']}{suffix}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report, path


def summary_lines(report: Details) -> list[str]:
    """The headline numbers, for the terminal and the PR description."""
    r = report["results"]
    lines = []
    if a := r.get("phase-a"):
        w = a["workload"]
        lines.append(
            f"Phase A: load {a['load_s']['total']} s · audio RTF {w['audio']['real_time_factor']} · "
            f"page {w['docling']['page_s']} s · chart2csv {w['vision']['chart2csv']['seconds']} s · "
            f"tables_html {w['vision']['tables_html']['seconds']} s · "
            f"embed {w['embed_chunks']['chunks_per_s']} chunks/s · peak {a['memory']['peak_footprint_gb']} GB"
        )
    if b := r.get("phase-b"):
        s, ret = b["server"], b["retrieval"]
        lines.append(
            f"Phase B: server ready {s['ready_s']} s + warm-up {s['warm_up_s']} s · "
            f"decode {b['decode_tps_short_context']} tok/s · peak {b['peak_footprint_gb']} GB"
        )
        for c in b["contexts"]:
            lines.append(
                f"  {c['prompt_tokens']:>6} prompt tokens: TTFT {c['ttft_s']} s "
                f"({c['prefill_tps']} tok/s prefill) · decode {c['decode_tps']} tok/s · "
                f"server {c['server_footprint_gb']} GB"
            )
        lines.append(
            f"  query embed p50 {ret['query_embed']['p50_ms']} ms · rerank 30 pairs p50 "
            f"{ret['rerank_30_pairs']['p50_ms']} ms"
        )
        for size, v in ret["vector_search_top50"].items():
            lines.append(
                f"  vectors {size}: numpy fp16 {v['numpy_fp16']['p50_ms']} ms · numpy fp32 "
                f"{v['numpy_fp32']['p50_ms']} ms · mps fp16 {v['torch_mps_fp16']['p50_ms']} ms"
            )
    if pc := r.get("prompt-cache"):
        for label, v in pc.items():
            lines.append(
                f"Prompt cache {label}: server GB after each request {v['footprint_after_each_gb']}"
            )
    if lc := r.get("long-context"):
        for label, v in lc.items():
            lines.append(
                f"Long context {label}: {v['prompt_tokens']} tokens, TTFT {v['ttft_s']} s "
                f"({v['prefill_tps']} tok/s), server peak {v['peak_footprint_gb']} GB"
            )
    if c := r.get("phase-c"):
        k = c["check_8_chunks"]
        lines.append(
            f"Phase C: load {c['load_s']['guardian']} s · check ({k['prompt_tokens']} tokens) p50 "
            f"{k['p50_ms'] / 1000:.2f} s · {c['check_24_chunks']['prompt_tokens']} tokens "
            f"{c['check_24_chunks']['seconds']} s · peak {c['memory']['peak_footprint_gb']} GB"
        )
    if sw := r.get("switches"):
        for i, cycle in enumerate(sw["cycles"], 1):
            lines.append(f"Switch cycle {i}: B→A {cycle['b_to_a_s']} s · A→B {cycle['a_to_b_s']} s")
    return lines + [c["line"] for c in report.get("budget", [])]
