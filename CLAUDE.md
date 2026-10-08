# granit

Local document & meeting intelligence on Apple Silicon (M2 Max, 32 GB) with IBM Granite models on MLX.
**PLAN.md is the source of truth**: §7 Decisions for choices, §5 for milestones. The current milestone is in `.claude/milestone`.

## Commands

```bash
uv sync                                   # install (uv-managed Python 3.12; never pip)
uv run pytest                             # unit tests only (default: -m "not model"), seconds
uv run pytest -m model                    # model smoke/golden tests on the Mac (loads GBs; one phase at a time)
uv run ruff format && uv run ruff check   # format + lint
uv run ty check src tests app             # types
uv run granit models list|download|convert|smoke
uv run granit transcribe FILE… [--json DIR]  # audio → timestamped segments (M2)
uv run granit convert FILE… [--out DIR] [--vision-tables]  # PDF/image → Markdown + chart data (M3)
uv run granit extract FILE --schema S.json  # form fields via Granite Vision, validated (M3)
uv run granit ingest FILE… [--data DIR]  # add to the library + run the Phase A worker (M4)
uv run granit search "question" [--mode bm25|vectors|hybrid|hybrid+rerank] [--json]  # cited hits + trace (M4)
uv run granit sources  # the library
uv run granit ask "question" [--thinking off|low|on]  # cited answer (M5; starts Q&A if needed)
uv run granit meeting SOURCE  # meeting summary JSON (M5)
uv run granit verify [--data DIR]  # Guardian checks unverified answers + summaries (M8, Phase C; not while Q&A runs)
uv run granit ui [--data DIR]  # the app: Ingest, Library, Ask, Extract (M6)
uv run --group screenshots python scripts/ui_screenshots.py --data DIR  # §4.8 layout check (24 screenshots)
uv run granit eval run --set public|private [--no-judge] [--fresh]  # quality eval (M7, ~15 min); results in eval/results/
uv run granit eval compare A.json B.json  # flags regressions ≥ 2 points; run before merging model/prompt/retrieval changes
uv run granit eval run --set public --fresh  # needed after ingest/chunking/ASR changes or regenerating eval/public (every file's sha changes)
cp -R data/eval/libraries/private $SCRATCH/lib && uv run granit meeting 1 --data $SCRATCH/lib  # experiment on a copy, never the eval library
uv run python scripts/make_eval_fixtures.py  # regenerate the public eval set (never hand-edit eval/public/)
uv run granit bench [SCENARIO…] [--quick]  # M1 benchmarks (~15 min); results in bench/results/
```

Quality gate before finishing (also hook H5 and CI): `ruff format --check`, `ruff check`, `ty check src tests app`, `pytest -m "not model"`.

## Rules

- **Dependencies:** `uv add` / `uv remove` only; never edit `uv.lock`. Pinned ML stack (PLAN.md §3.4): the transformers window is narrow, so upgrade deliberately.
- **Licensing:** Apache-2.0/MIT/OpenMDW-1.1 models only (not the revocable NVIDIA Open Model License, PLAN.md §3.5), listed in `src/granit/config.py` and approved in `tests/unit/test_config.py`. Never the `-nc` TurboCTC variant. New dependencies must pass `test_dependency_licenses.py`; exceptions go in `licenses_overrides.toml` with a reason.
- **Pin models by commit SHA** in `config.py`. Load from the local snapshot (`granit.models.download.local_snapshot`), never by bare repo ID.
- **Lazy imports:** `mlx*`, `torch`, `sentence_transformers`, `transformers`, `docling` are imported inside functions only (`test_lazy_imports.py`). Unit tests use fakes; no weights or GPU in CI (`HF_HUB_OFFLINE=1`).
- **One phase at a time** (PLAN.md §2.1): ingest (A), Q&A (B) or verify (C). Never load models while `mlx_lm.server` or a worker runs (hook H4). Memory is freed by ending processes.
- **Phase B memory limits:** start the LLM server via `granit.models.server.ServerConfig` (2 GB prompt-cache cap + 1 GB MLX cache limit, PLAN.md §3.3); mlx-lm's defaults exceed the GPU limit at long contexts.
- **UI:** page scripts in `app/app_pages/` stay thin; logic goes in `src/granit/ui/` (tested). Every Markdown-rendering call escapes what it interpolates (`safe_md`; `test_layout.py` checks). Pages open a store per run (`open_store()`); never share a connection across reruns.
- **Phases:** the UI talks to the LLM only through `PhaseManager` (`models/phases.py`): wrap each answer in `phases.chat()`, call `tick()` periodically, `process_now()` for the button. Never start `mlx_lm.server` or the worker directly from UI code.
- **No models in the Streamlit process**, except the query embedder + reranker behind a lock (PLAN.md §2.2).
- **Never touch** `data/` (user data) or `models/` (weights). Both are gitignored and protected by hook H2.
- **Releases come only from CI** (PLAN.md §4.3): bump with `uv version --bump …`, merge a PR. No local tags or `gh release`.
- **Prompt A/B:** run each variant in its own fresh process (`cli._llm_server()` + patched prompt): `mlx_lm.server`'s prompt cache makes temperature-0 output depend on earlier requests (same prompt 0.467 vs 0.383 by order).
- **Stacked PRs:** `gh pr merge --delete-branch` on the base closes the PR stacked on it (no retarget). Base PRs on `main`, or merge `main` into the stacked branch and open a new PR.
- **Model tests after long eval runs:** the answer-latency checks (`MAX_ANSWER_S`) can fail under load; re-run `tests/models/test_reasoning.py` alone before suspecting the change.
- `config.py` and `.claude/hooks/*` are standard library only (hooks run with the system `python3`, 3.11+).

## Layout

`src/granit/` package (`config.py`, `cli.py`, `models/`, later `ingest/`, `store/`, `search/`, `reason/`, `verify/`, `evaluate/`) ·
`app/` Streamlit (M6) · `tests/unit/` no models · `tests/models/` `@pytest.mark.model` · `.claude/hooks/` H1–H6 (PLAN.md §4.1).
