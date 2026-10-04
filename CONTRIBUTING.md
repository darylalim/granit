# Contributing to granit

## Setup

Requirements: an Apple Silicon Mac (32 GB recommended), macOS 26, [uv](https://docs.astral.sh/uv/).

```bash
uv sync                              # creates .venv with uv-managed Python 3.12
uv run granit models download        # ~20 GB of pinned model weights into ~/.cache/huggingface
uv run granit models convert --delete-source   # Guardian 4.1 8B → local 8-bit MLX (needs ~17 GB temporarily)
uv run granit models smoke           # load each model once and run a known-answer check
```

Use `uv add` / `uv remove` for dependencies. Don't use pip and don't edit `uv.lock` by hand.

## Quality checks

The same checks run in CI, and in Claude Code's Stop hook:

```bash
uv run ruff format --check && uv run ruff check && uv run ty check src tests && uv run pytest
```

`uv run pytest` runs unit tests only (`-m "not model"`). These need no weights or GPU and take seconds.
Model tests run on the Mac with `uv run pytest -m model`. Close the app first, because only one phase may hold
models at a time.

## Licensing

granit is Apache-2.0. Contributions are accepted under the same license (Apache-2.0 §5), so no CLA is needed.
- New dependencies must pass `tests/unit/test_dependency_licenses.py` (permissive licenses plus MPL-2.0; no GPL, AGPL,
  SSPL, non-commercial or unknown licenses). Hand-reviewed exceptions go in `licenses_overrides.toml` with a reason.
- New models must be Apache-2.0 or MIT. Add them to `src/granit/config.py` (pinned to a commit SHA) **and** to the
  approved list in `tests/unit/test_config.py`.

## Pull requests

`main` is protected: every change goes through a PR, and the `lint` and `test` checks must pass.
PRs are squash-merged.

## How to cut a release

Releases are automatic: when the version in `pyproject.toml` has no matching `v<version>` tag, a green build on `main`
is tagged and published as a GitHub Release (wheel, sdist, third-party notices).

```bash
uv version --bump minor              # or patch / major; prerelease: --bump minor --bump beta
git commit -am "Release 0.2.0"       # commits pyproject.toml + uv.lock together
# open a PR → merge → CI publishes the release
```

Never create tags or releases by hand.
