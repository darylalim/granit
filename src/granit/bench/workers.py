"""Benchmark worker processes: ``python -m granit.bench.workers <phase-a|retrieval|guardian> [--hold] [--quick]``.

Each worker loads one phase's models, runs a representative workload and prints ``RESULT <json>`` as its last
line. ``--hold`` (Phase A, retrieval) prints ``READY`` after loading and then waits for ``exit`` on stdin, so
the orchestrator can time phase switches and measure memory while another process is busy.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from granit.bench.stats import summarize, timed
from granit.config import EMBEDDING_DIM, HUB_MODELS, LOCAL_MODELS
from granit.fixtures import say_wav, synthetic_chunks, synthetic_text, text_page_png
from granit.models.download import local_snapshot
from granit.models.memory import footprint, mlx_peak_gb, mps_allocated_gb

Details = dict[str, Any]
SAMPLE_RATE = 16_000
MAX_CHUNK_S = 30.0  # PLAN.md §3.5: chunks of at most ~30 s


def _load_timed(loads: dict[str, float], name: str, fn: Callable[[], Any]) -> Any:
    start = time.perf_counter()
    value = fn()
    loads[name] = round(time.perf_counter() - start, 2)
    return value


def _hold() -> None:
    """Handshake with the orchestrator: report READY, then wait for `exit` (or EOF)."""
    print("READY", flush=True)
    for line in sys.stdin:
        if line.strip() == "exit":
            break


def _memory(**extra: float) -> Details:
    fp = footprint()
    return {
        "peak_footprint_gb": round(fp.peak_gb, 2),
        "current_footprint_gb": round(fp.current_gb, 2),
        **{k: round(v, 2) for k, v in extra.items()},
    }


# ── Phase A: ingest worker (Speech + VAD + Docling + Vision bf16 + Embedding) ──


def load_embedder() -> Any:
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        str(local_snapshot(HUB_MODELS["embedding"])),
        device="mps",
        model_kwargs={"torch_dtype": torch.float16},
    )


def load_reranker() -> Any:
    import torch
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        str(local_snapshot(HUB_MODELS["reranker"])),
        device="mps",
        model_kwargs={"torch_dtype": torch.float16},
    )


def load_docling() -> Any:
    from docling.datamodel import vlm_model_specs
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.document_converter import DocumentConverter, ImageFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    model = HUB_MODELS["docling"]
    local_snapshot(model)
    options = VlmPipelineOptions(
        vlm_options=vlm_model_specs.GRANITEDOCLING_MLX.model_copy(
            update={"repo_id": model.repo_id, "revision": model.revision}
        )
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(pipeline_cls=VlmPipeline, pipeline_options=options)
        }
    )
    converter.initialize_pipeline(InputFormat.IMAGE)  # load the model now, not on the first page
    return converter


def phase_a(hold: bool, quick: bool) -> Details:
    import mlx_audio.vad as vad_lib
    from mlx_audio.stt.utils import load_model
    from mlx_vlm import load as load_vlm

    loads: dict[str, float] = {}
    start = time.perf_counter()
    speech = _load_timed(loads, "speech", lambda: load_model(local_snapshot(HUB_MODELS["speech"])))
    vad = _load_timed(loads, "vad", lambda: vad_lib.load(local_snapshot(HUB_MODELS["vad"])))
    converter = _load_timed(loads, "docling", load_docling)
    vision_path = local_snapshot(HUB_MODELS["vision"])
    vision_model, vision_processor = _load_timed(
        loads, "vision", lambda: load_vlm(str(vision_path))
    )
    embedder = _load_timed(loads, "embedding", load_embedder)
    loads["total"] = round(time.perf_counter() - start, 2)
    after_load = _memory()
    if hold:
        _hold()
        return {"load_s": loads, "memory_after_load": after_load}

    work: Details = {}
    stage_memory: Details = {}  # footprint after each stage: which one drives the phase's peak?

    def stage(name: str, fn: Callable[[], Details]) -> None:
        work[name] = fn()
        fp = footprint()
        stage_memory[name] = {
            "current_gb": round(fp.current_gb, 2),
            "peak_gb": round(fp.peak_gb, 2),
        }

    with tempfile.TemporaryDirectory() as tmp:
        stage("audio", lambda: _audio_workload(speech, vad, Path(tmp), seconds=20 if quick else 60))
        stage("docling", lambda: _docling_workload(converter, Path(tmp)))
    stage(
        "vision",
        lambda: _vision_workload(
            vision_model, vision_processor, vision_path, max_tokens=300 if quick else 800
        ),
    )
    stage("embed_chunks", lambda: _embed_workload(embedder, n=64 if quick else 256))
    return {
        "load_s": loads,
        "memory_after_load": after_load,
        "workload": work,
        "memory_by_stage": stage_memory,
        "memory": _memory(mlx_peak_gb=mlx_peak_gb(), mps_gb=mps_allocated_gb()),
    }


def _audio_workload(speech: Any, vad: Any, tmp: Path, seconds: int) -> Details:
    import miniaudio
    import numpy as np

    # ~15 characters of speech per second with macOS `say` at its default rate.
    wav = say_wav(synthetic_text(seconds * 15, seed=7), tmp, "meeting")
    decoded = miniaudio.decode_file(
        str(wav), output_format=miniaudio.SampleFormat.FLOAT32, nchannels=1, sample_rate=SAMPLE_RATE
    )
    audio = np.frombuffer(decoded.samples, dtype=np.float32)
    duration = len(audio) / SAMPLE_RATE

    start = time.perf_counter()
    segments = vad.get_speech_timestamps(audio, sample_rate=SAMPLE_RATE, return_seconds=True)
    vad_s = time.perf_counter() - start

    # Merge speech segments greedily into chunks of at most MAX_CHUNK_S (M2 refines this).
    chunks: list[tuple[float, float]] = []
    for seg in segments:
        if chunks and seg["end"] - chunks[-1][0] <= MAX_CHUNK_S:
            chunks[-1] = (chunks[-1][0], seg["end"])
        else:
            chunks.append((seg["start"], seg["end"]))
    start = time.perf_counter()
    words = 0
    for begin, end in chunks:
        piece = audio[int(begin * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
        words += len(speech.generate(piece).text.split())
    asr_s = time.perf_counter() - start
    return {
        "audio_s": round(duration, 1),
        "vad_s": round(vad_s, 3),
        "chunks": len(chunks),
        "transcribe_s": round(asr_s, 2),
        "real_time_factor": round((vad_s + asr_s) / duration, 3),
        "words": words,
    }


def _docling_workload(converter: Any, tmp: Path) -> Details:
    lines = tuple(synthetic_text(1400, seed=3).replace("\n\n", " ").split(". "))[:18]
    page = text_page_png(tuple(line[:60] for line in lines), tmp / "page.png")
    start = time.perf_counter()
    markdown = converter.convert(page).document.export_to_markdown()
    return {"page_s": round(time.perf_counter() - start, 2), "markdown_chars": len(markdown)}


def _vision_workload(model: Any, processor: Any, path: Path, max_tokens: int) -> Details:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    out: Details = {}
    for tag, image in (("<chart2csv>", "chart.jpg"), ("<tables_html>", "table.png")):
        prompt = apply_chat_template(processor, model.config, tag, num_images=1)
        assert isinstance(prompt, str)
        start = time.perf_counter()
        result = generate(
            model,
            processor,
            prompt,
            image=[str(path / image)],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        out[tag.strip("<>")] = {
            "seconds": round(time.perf_counter() - start, 2),
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
            "prompt_tps": round(result.prompt_tps, 1),
            "generation_tps": round(result.generation_tps, 1),
        }
    return out


def _embed_workload(embedder: Any, n: int) -> Details:
    chunks = synthetic_chunks(n, seed=11)
    embedder.encode(chunks[:8], normalize_embeddings=True)  # warm-up
    start = time.perf_counter()
    vectors = embedder.encode(chunks, batch_size=32, normalize_embeddings=True)
    seconds = time.perf_counter() - start
    assert vectors.shape == (n, EMBEDDING_DIM)
    return {"chunks": n, "seconds": round(seconds, 2), "chunks_per_s": round(n / seconds, 1)}


# ── Phase B: query embedder + reranker in the UI backend, plus the vector matrix (§2.2, §2.3) ──


def retrieval(hold: bool, quick: bool) -> Details:

    loads: dict[str, float] = {}
    embedder = _load_timed(loads, "embedding", load_embedder)
    reranker = _load_timed(loads, "reranker", load_reranker)

    query = "Which invoices did Finance approve for more than $10,000 before Friday?"
    query_embed = summarize(
        timed(lambda: embedder.encode([query], normalize_embeddings=True), repeat=30)
    )
    # PLAN.md §3.2: the reranker scores the RRF top ~30 chunks (~300 tokens each) for one question.
    pairs = synthetic_chunks(30, seed=5)
    rerank_30 = summarize(timed(lambda: reranker.rank(query, pairs), repeat=5 if quick else 10))

    peak_models_gb = footprint().peak_gb  # embedder + reranker after their workloads
    query_vec = embedder.encode([query], normalize_embeddings=True)[0]
    # Keep the 150K matrix on MPS while held: that's the realistic Phase B state alongside the server.
    vectors_150k, matrix_150k = _vector_search(query_vec, 150_000)
    with_150k_gb = footprint().current_gb
    if hold:
        _hold()
    del matrix_150k
    # The 1M what-if (sqlite-vec trigger, §2.3) runs after release, when nothing else is loaded.
    vectors_1m, _ = _vector_search(query_vec, 1_000_000)
    return {
        "load_s": loads,
        "query_embed": query_embed,
        "rerank_30_pairs": rerank_30,
        "vector_search_top50": {"150k": vectors_150k, "1000k": vectors_1m},
        "memory": {
            "peak_models_gb": round(peak_models_gb, 2),
            # Phase B's retrieval share at the plan's 150K-chunk reference size, matrix on MPS.
            "peak_with_150k_matrix_gb": round(max(peak_models_gb, with_150k_gb), 2),
            "mps_gb": round(mps_allocated_gb(), 2),
        },
    }


def _vector_search(query: Any, n: int) -> tuple[Details, Any]:
    """Top-50 by dot product over ``n`` normalized fp16 vectors (§2.3), three ways.

    NumPy has no BLAS path for float16, so ``fp16 @ fp16`` runs generic loops; fp32 math or the GPU are the
    alternatives. Returns the timings and the matrix as an MPS tensor (the NumPy copies are freed).
    """
    import gc

    import numpy as np
    import torch

    rng = np.random.default_rng(n)
    q16, q32 = query.astype(np.float16), query.astype(np.float32)
    q_mps = torch.from_numpy(q16).to("mps")
    m32 = rng.standard_normal((n, EMBEDDING_DIM), dtype=np.float32)
    m32 /= np.linalg.norm(m32, axis=1, keepdims=True)
    m16 = m32.astype(np.float16)

    def top50(scores: Any) -> Any:
        top = np.argpartition(scores, -50)[-50:]
        return top[np.argsort(scores[top])[::-1]]

    numpy_fp32 = summarize(timed(lambda m=m32: top50(m @ q32), repeat=20))
    del m32
    numpy_fp16 = summarize(timed(lambda m=m16: top50(m @ q16), repeat=5))
    m_mps = torch.from_numpy(m16).to("mps")
    del m16
    gc.collect()
    mps_fp16 = summarize(timed(lambda m=m_mps: torch.topk(m @ q_mps, 50).indices.cpu(), repeat=20))
    torch.mps.synchronize()
    details = {
        "matrix_fp16_mb": round(n * EMBEDDING_DIM * 2 / 1e6, 1),
        "numpy_fp16": numpy_fp16,
        "numpy_fp32": numpy_fp32,
        "torch_mps_fp16": mps_fp16,
    }
    return details, m_mps


# ── Phase C: Guardian 4.1 8B q8, no-think groundedness checks (§2.4) ──


def guardian(hold: bool, quick: bool) -> Details:
    from mlx_lm import load, stream_generate

    from granit.verify.guardian import groundedness_messages, parse_score

    del hold  # nothing runs alongside Phase C
    model_spec = LOCAL_MODELS["guardian"]
    loads: dict[str, float] = {}
    model, tokenizer = _load_timed(loads, "guardian", lambda: load(str(model_spec.path))[:2])

    def check(n_chunks: int, seed: int) -> Details:
        documents = [
            {"doc_id": str(i), "text": chunk}
            for i, chunk in enumerate(synthetic_chunks(n_chunks, seed=seed))
        ]
        answer = synthetic_text(400, seed=seed + 1)
        prompt = tokenizer.apply_chat_template(
            groundedness_messages(answer, "What are the open action items?"),
            tokenize=False,
            add_generation_prompt=True,
            documents=documents,
        )
        start = time.perf_counter()
        text, last = "", None
        for response in stream_generate(model, tokenizer, prompt, max_tokens=24):
            text += response.text
            last = response
        assert last is not None
        return {
            "seconds": time.perf_counter() - start,
            "prompt_tokens": last.prompt_tokens,
            "prompt_tps": last.prompt_tps,
            "score": parse_score(text),
        }

    # A RAG answer with its 8 cited chunks (~2.5K tokens), then one near the 8K Phase C context.
    checks = [check(8, seed=100 + i) for i in range(3 if quick else 10)]
    long_check = check(24, seed=999)
    return {
        "load_s": loads,
        "check_8_chunks": {
            **summarize([c["seconds"] for c in checks]),
            "prompt_tokens": checks[0]["prompt_tokens"],
            "prompt_tps": round(sum(c["prompt_tps"] for c in checks) / len(checks), 1),
            "parsed_scores": sum(c["score"] in {"yes", "no"} for c in checks),
        },
        "check_24_chunks": {
            "seconds": round(long_check["seconds"], 2),
            "prompt_tokens": long_check["prompt_tokens"],
            "score": long_check["score"],
        },
        "memory": _memory(mlx_peak_gb=mlx_peak_gb()),
    }


WORKERS: dict[str, Callable[[bool, bool], Details]] = {
    "phase-a": phase_a,
    "retrieval": retrieval,
    "guardian": guardian,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m granit.bench.workers")
    parser.add_argument("worker", choices=sorted(WORKERS))
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    result = WORKERS[args.worker](args.hold, args.quick)
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
