# granit

Local document & meeting intelligence on Apple Silicon (M2 Max, 32 GB) with IBM Granite models on MLX.
**PLAN.md is the source of truth**: §7 Decisions for choices, §5 for milestones. The current milestone is in `.claude/milestone`.

## Commands

```bash
uv sync                                   # install (uv-managed Python 3.12; never pip)
uv run pytest                             # unit tests only (default: -m "not model"), seconds
uv run pytest -m model                    # model smoke/golden tests on the Mac (loads GBs; one phase at a time)
uv run ruff format && uv run ruff check   # format + lint
uv run ty check src tests                 # types
uv run granit models list|download|convert|smoke
uv run granit transcribe FILE… [--json DIR]  # audio → timestamped segments (M2)
uv run granit convert FILE… [--out DIR] [--vision-tables]  # PDF/image → Markdown + chart data (M3)
uv run granit extract FILE --schema S.json  # form fields via Granite Vision, validated (M3)
uv run granit bench [SCENARIO…] [--quick]  # M1 benchmarks (~15 min); results in bench/results/
```

Quality gate before finishing (also hook H5 and CI): `ruff format --check`, `ruff check`, `ty check src tests`, `pytest -m "not model"`.

## Rules

- **Dependencies:** `uv add` / `uv remove` only; never edit `uv.lock`. Pinned ML stack (PLAN.md §3.4): the transformers window is narrow, so upgrade deliberately.
- **Licensing:** Apache-2.0/MIT models only, listed in `src/granit/config.py` and approved in `tests/unit/test_config.py`. Never the `-nc` TurboCTC variant. New dependencies must pass `test_dependency_licenses.py`; exceptions go in `licenses_overrides.toml` with a reason.
- **Pin models by commit SHA** in `config.py`. Load from the local snapshot (`granit.models.download.local_snapshot`), never by bare repo ID.
- **Lazy imports:** `mlx*`, `torch`, `sentence_transformers`, `transformers`, `docling` are imported inside functions only (`test_lazy_imports.py`). Unit tests use fakes; no weights or GPU in CI (`HF_HUB_OFFLINE=1`).
- **One phase at a time** (PLAN.md §2.1): ingest (A), Q&A (B) or verify (C). Never load models while `mlx_lm.server` or a worker runs (hook H4). Memory is freed by ending processes.
- **Phase B memory limits:** start the LLM server via `granit.models.server.ServerConfig` (2 GB prompt-cache cap + 1 GB MLX cache limit, PLAN.md §3.3); mlx-lm's defaults exceed the GPU limit at long contexts.
- **No models in the Streamlit process**, except the query embedder + reranker behind a lock (PLAN.md §2.2).
- **Never touch** `data/` (user data) or `models/` (weights). Both are gitignored and protected by hook H2.
- **Releases come only from CI** (PLAN.md §4.3): bump with `uv version --bump …`, merge a PR. No local tags or `gh release`.
- `config.py` and `.claude/hooks/*` are standard library only (hooks run with the system `python3`, 3.11+).

## Layout

`src/granit/` package (`config.py`, `cli.py`, `models/`, later `ingest/`, `store/`, `search/`, `reason/`, `verify/`, `evaluate/`) ·
`app/` Streamlit (M6) · `tests/unit/` no models · `tests/models/` `@pytest.mark.model` · `.claude/hooks/` H1–H6 (PLAN.md §4.1).
