"""M0 smoke checks: load each model at its pinned revision and run one tiny, known-answer task.

Every check runs in its **own process** (``python -m granit.models.smoke <key>``) with ``HF_HUB_OFFLINE=1``:
memory is fully returned when the process exits (PLAN.md §2.1), and a missing file fails instead of downloading.
These are smoke checks, not golden tests: thresholds and fixtures for those arrive with M2–M7.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from granit.config import HUB_MODELS, LOCAL_MODELS
from granit.fixtures import DIALOGUE, dialogue_wav, say_wav, text_page_png
from granit.models.download import local_snapshot
from granit.models.memory import footprint, mlx_peak_gb, mps_allocated_gb
from granit.verify.guardian import groundedness_messages, parse_score

Details = dict[str, Any]

SPEECH_TEXT = (
    "Invoice number forty two is due on Friday. Please send the revised invoice to Finance."
)
DOC_LINES = ("Quarterly Report", "Invoice INV-2026-0042 has a total of 4,980 dollars.")
EMBED_QUERY = "What is the total on invoice INV-2026-0042?"
EMBED_PASSAGES = (
    "Invoice INV-2026-0042: the total due is $4,980, payable by Friday.",
    "The quarterly planning meeting moved to Tuesday afternoon.",
    "Granite is an igneous rock made mostly of quartz and feldspar.",
)
# From the reranker's model card: the two H2O passages must rank first.
RERANK_QUERY = "what is the chemical formula of water?"
RERANK_PASSAGES = (
    "Romeo and Juliet is a play by William Shakespeare.",
    "Climate change refers to long-term shifts in temperatures.",
    "Shakespeare also wrote Hamlet and Macbeth.",
    "Water is an inorganic compound with the chemical formula H2O.",
    "In liquid form, H2O is also called 'water' at standard temperature and pressure.",
)
# From the Guardian model card (Example 2), run in no-think mode: the answer contradicts the document → "yes".
GUARDIAN_DOCUMENT = (
    "Eat (1964) is a 45-minute underground film created by Andy Warhol and featuring painter Robert "
    "Indiana, filmed on Sunday, February 2, 1964, in Indiana's studio. The film was first shown by Jonas "
    "Mekas on July 16, 1964, at the Washington Square Gallery at 530 West Broadway.\nJonas Mekas "
    "(December 24, 1922 – January 23, 2019) was a Lithuanian-American filmmaker, poet, and artist who has "
    'been called "the godfather of American avant-garde cinema".'
)
GUARDIAN_ANSWER = (
    "The film Eat was first shown by Jonas Mekas on December 24, 1922 at the Washington Square Gallery "
    "at 530 West Broadway."
)


class SmokeFailure(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


# ── checks: each returns details for the report and raises SmokeFailure on a wrong answer ──


def check_speech() -> Details:
    from mlx_audio.stt.utils import load_model

    model = load_model(local_snapshot(HUB_MODELS["speech"]))
    with tempfile.TemporaryDirectory() as tmp:
        text = model.generate(str(say_wav(SPEECH_TEXT, Path(tmp)))).text.lower()
    for word in ("invoice", "friday", "finance"):
        expect(word in text, f"expected {word!r} in transcript {text!r}")
    return {"transcript": text, "mlx_peak_gb": round(mlx_peak_gb(), 2)}


def check_vad() -> Details:
    import mlx_audio.vad as vad

    model = vad.load(local_snapshot(HUB_MODELS["vad"]))
    with tempfile.TemporaryDirectory() as tmp:
        # `say` silence command: two phrases separated by 2 s of silence.
        wav = say_wav("First phrase here. [[slnc 2000]] Second phrase here.", Path(tmp), "vad")
        segments = model.get_speech_timestamps(str(wav), return_seconds=True)
    expect(len(segments) == 2, f"expected 2 speech segments, got {segments}")
    gap = segments[1]["start"] - segments[0]["end"]
    expect(gap > 1.0, f"expected a pause > 1 s between segments, got {gap:.2f} s")
    return {"segments": segments}


def check_docling() -> Details:
    from docling.datamodel import vlm_model_specs
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.document_converter import DocumentConverter, ImageFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    model = HUB_MODELS["docling"]
    local_snapshot(model)  # fail early with a clear message if it's not downloaded
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
    with tempfile.TemporaryDirectory() as tmp:
        page = text_page_png(DOC_LINES, Path(tmp) / "page.png")
        markdown = converter.convert(page).document.export_to_markdown()
    expect("Quarterly Report" in markdown, f"heading missing from {markdown!r}")
    expect("INV-2026-0042" in markdown, f"invoice ID missing from {markdown!r}")
    return {"markdown": markdown, "mlx_peak_gb": round(mlx_peak_gb(), 2)}


def check_vision() -> Details:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    path = local_snapshot(HUB_MODELS["vision"])
    model, processor = load(str(path))
    # The card's chart; the task tag is expanded by the model's own chat template (PLAN.md §3.1).
    prompt = apply_chat_template(processor, model.config, "<chart2csv>", num_images=1)
    assert isinstance(prompt, str)
    result = generate(
        model,
        cast(
            Any, processor
        ),  # mlx-vlm's ProcessorLike protocol is narrower than what load() returns
        prompt,
        image=[str(path / "chart.jpg")],
        max_tokens=400,
        temperature=0.0,
    )
    rows = [line for line in result.text.strip().splitlines() if "," in line]
    expect(len(rows) >= 2, f"expected CSV rows, got {result.text!r}")
    return {"csv_head": rows[:4], "mlx_peak_gb": round(mlx_peak_gb(), 2)}


def check_llm() -> Details:
    from mlx_lm import generate, load

    model, tokenizer = load(str(local_snapshot(HUB_MODELS["llm"])))[:2]
    messages = [{"role": "user", "content": "What is 17 + 25? Reply with the number only."}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    text = generate(model, tokenizer, prompt, max_tokens=256)
    expect("42" in text, f"expected 42 in {text!r}")
    return {"answer": text.strip()[-200:], "mlx_peak_gb": round(mlx_peak_gb(), 2)}


def load_embedder() -> Any:
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        str(local_snapshot(HUB_MODELS["embedding"])),
        device="mps",
        model_kwargs={"torch_dtype": torch.float16},
    )


def check_embedding() -> Details:
    import numpy as np

    model = load_embedder()
    vectors = model.encode([EMBED_QUERY, *EMBED_PASSAGES], normalize_embeddings=True)
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
    expect(vectors.shape[1] == 768, f"expected 768-d vectors, got {vectors.shape}")
    expect(bool(np.allclose(norms, 1.0, atol=1e-2)), f"vectors not normalized: {norms}")
    scores = vectors[1:] @ vectors[0]
    expect(int(np.argmax(scores)) == 0, f"invoice passage should rank first, scores {scores}")
    return {"scores": [round(float(s), 3) for s in scores], "mps_gb": round(mps_allocated_gb(), 2)}


def check_reranker() -> Details:
    import torch
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        str(local_snapshot(HUB_MODELS["reranker"])),
        device="mps",
        model_kwargs={"torch_dtype": torch.float16},
    )
    ranks = model.rank(RERANK_QUERY, list(RERANK_PASSAGES))
    top_two = {r["corpus_id"] for r in ranks[:2]}
    expect(top_two == {3, 4}, f"H2O passages should rank first, got {ranks}")
    return {
        "ranking": [(int(r["corpus_id"]), round(float(r["score"]), 3)) for r in ranks],
        "mps_gb": round(mps_allocated_gb(), 2),
    }


def check_mps_threads() -> Details:
    """PLAN.md §2.2: the query embedder must behave when called from Streamlit-style threads.

    Streamlit runs each script rerun in a fresh thread, so each call here runs in a new
    ``threading.Thread`` (not a pool), all sharing one model behind a lock.
    """
    import numpy as np

    model = load_embedder()
    texts = [f"{EMBED_QUERY} (variant {i})" for i in range(8)]
    reference = model.encode(texts, normalize_embeddings=True)
    lock = threading.Lock()
    results: dict[int, Any] = {}
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            with lock:
                results[i] = model.encode([texts[i % 8]], normalize_embeddings=True)[0]
        except BaseException as exc:  # reported below, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    expect(not errors, f"thread errors: {errors!r}")
    expect(len(results) == 32, f"only {len(results)}/32 thread calls finished")
    max_diff = max(
        float(np.abs(results[i].astype(np.float32) - reference[i % 8].astype(np.float32)).max())
        for i in results
    )
    expect(max_diff < 1e-2, f"threaded embeddings differ from serial ones by {max_diff}")
    return {"thread_calls": len(results), "max_abs_diff": max_diff}


def check_guardian() -> Details:
    from mlx_lm import generate, load

    guardian = LOCAL_MODELS["guardian"]
    expect(guardian.is_built(), "Guardian q8 is not built; run: uv run granit models convert")
    model, tokenizer = load(str(guardian.path))[:2]
    messages = groundedness_messages(GUARDIAN_ANSWER)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        documents=[{"doc_id": "0", "text": GUARDIAN_DOCUMENT}],
    )
    text = generate(model, tokenizer, prompt, max_tokens=64)
    score = parse_score(text)
    expect(score == "yes", f"expected <score>yes</score> (ungrounded), got {text!r}")
    return {"raw": text.strip(), "mlx_peak_gb": round(mlx_peak_gb(), 2)}


def check_diarization() -> Details:
    from granit.ingest.audio import decode
    from granit.ingest.speakers import Diarizer

    diarizer = Diarizer()
    expect(
        diarizer.available(),
        "the diarization model is not built; run: uv run granit models convert diarization",
    )
    with tempfile.TemporaryDirectory() as tmp:
        wav, lines = dialogue_wav(DIALOGUE, Path(tmp))
        probs, frame_s = diarizer.probabilities(decode(wav))
    # Each line's dominant channel: Daniel's two lines share one, Samantha's two another.
    channels = [
        int(probs[int(s / frame_s) : int(e / frame_s)].sum(axis=0).argmax()) for s, e, _ in lines
    ]
    expect(
        channels[0] == channels[2] != channels[1] == channels[3],
        f"expected two alternating speakers, got channels {channels}",
    )
    return {"channels": channels, "mlx_peak_gb": round(mlx_peak_gb(), 2)}


CHECKS: dict[str, Callable[[], Details]] = {
    "speech": check_speech,
    "vad": check_vad,
    "docling": check_docling,
    "vision": check_vision,
    "llm": check_llm,
    "embedding": check_embedding,
    "reranker": check_reranker,
    "mps-threads": check_mps_threads,
    "guardian": check_guardian,
    "diarization": check_diarization,
}


# ── running ──


@dataclass
class SmokeResult:
    key: str
    ok: bool
    seconds: float = 0.0
    peak_footprint_gb: float = 0.0
    details: Details = field(default_factory=dict)
    error: str = ""

    def summary(self) -> str:
        mark = "✅" if self.ok else "❌"
        head = f"{mark} {self.key:<12} {self.seconds:6.1f} s  peak {self.peak_footprint_gb:5.2f} GB"
        body = json.dumps(self.details, ensure_ascii=False) if self.ok else self.error
        return f"{head}  {body}"


def run_in_process(key: str) -> SmokeResult:
    start = time.perf_counter()
    try:
        details = CHECKS[key]()
        ok, error = True, ""
    except Exception as exc:
        details, ok, error = {}, False, f"{type(exc).__name__}: {exc}"
    # Physical footprint includes Metal (GPU) buffers; RSS would miss most MLX / MPS memory.
    peak = footprint().peak_gb
    return SmokeResult(
        key, ok, round(time.perf_counter() - start, 1), round(peak, 2), details, error
    )


def run_isolated(key: str, timeout: float = 1800) -> SmokeResult:
    """Run one check in a fresh Python process, offline, and parse its JSON result line."""
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "granit.models.smoke", key],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return SmokeResult(key, False, timeout, error=f"timed out after {timeout:.0f} s")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            return SmokeResult(**json.loads(line))
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
    return SmokeResult(key, False, error=f"exit {proc.returncode}: " + " | ".join(tail))


if __name__ == "__main__":
    result = run_in_process(sys.argv[1])
    print(json.dumps(asdict(result), default=str))
    sys.exit(0 if result.ok else 1)
