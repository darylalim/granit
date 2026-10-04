"""Model management without models: download / convert / smoke plumbing with fakes."""

from __future__ import annotations

import json
import sys
import types
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest

from granit import config
from granit.models import convert, download, smoke


def fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> None:
    monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))


# ── download ──


def test_download_passes_pinned_revision_and_ignores(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def snapshot_download(repo_id: str, **kwargs: Any) -> str:
        calls.append((repo_id, kwargs))
        return f"/cache/{repo_id}"

    fake_module(monkeypatch, "huggingface_hub", snapshot_download=snapshot_download)
    model = config.HUB_MODELS["embedding"]
    assert download.download(model) == Path("/cache/ibm-granite/granite-embedding-english-r2")
    assert calls == [
        (model.repo_id, {"revision": model.revision, "ignore_patterns": ["pytorch_model.bin"]})
    ]
    download.download(config.HUB_MODELS["llm"])
    assert calls[-1][1]["ignore_patterns"] is None


def test_local_snapshot_explains_how_to_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="granit models download vision"):
        download.local_snapshot(config.HUB_MODELS["vision"])


def test_delete_cached_only_touches_the_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    Rev = namedtuple("Rev", "commit_hash")
    Repo = namedtuple("Repo", "repo_id revisions")
    deleted: list[str] = []

    class Strategy:
        expected_freed_size = 123

        def execute(self) -> None:
            deleted.append("executed")

    class Cache:
        def __init__(self) -> None:
            revisions = [Rev("other"), Rev("ab01ccca5dcfb80246369a086a4a87a29198f5af")]
            self.repos = [Repo("ibm-granite/granite-guardian-4.1-8b", revisions)]

        def delete_revisions(self, *revisions: str) -> Strategy:
            deleted.extend(revisions)
            return Strategy()

    fake_module(monkeypatch, "huggingface_hub", scan_cache_dir=Cache)
    source = config.HUB_MODELS["guardian-source"]
    assert download.delete_cached(source) == 123
    assert deleted == [source.revision, "executed"]
    assert download.delete_cached(config.HUB_MODELS["llm"]) == 0  # not cached: nothing to delete


# ── convert ──


def local_model(tmp_path: Path) -> config.LocalModel:
    return config.LocalModel("guardian", "guardian-source", tmp_path / "out", 8, "C", 0.001)


def fake_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    monkeypatch.setattr(convert, "local_snapshot", lambda _model: source)


def test_build_quantizes_and_records_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_source(monkeypatch, tmp_path)
    calls = []

    def fake_convert(**kwargs: Any) -> None:
        calls.append(kwargs)
        Path(kwargs["mlx_path"]).mkdir()
        (Path(kwargs["mlx_path"]) / "config.json").write_text("{}")

    fake_module(monkeypatch, "mlx_lm", convert=fake_convert)
    monkeypatch.setattr(convert, "version", lambda _pkg: "0.32.0")
    model = local_model(tmp_path)
    assert convert.build(model) == model.path
    assert calls == [
        {
            "hf_path": str(tmp_path / "src"),
            "mlx_path": str(model.path),
            "quantize": True,
            "q_bits": 8,
        }
    ]
    record = json.loads((model.path / config.SOURCE_RECORD).read_text())
    assert record["source_repo"] == "ibm-granite/granite-guardian-4.1-8b"
    assert record["source_revision"] == config.HUB_MODELS["guardian-source"].revision
    assert record["q_bits"] == 8
    assert model.is_built()

    convert.build(model)  # already built: no second conversion
    assert len(calls) == 1


def test_build_refuses_a_half_finished_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_source(monkeypatch, tmp_path)
    model = local_model(tmp_path)
    model.path.mkdir()
    (model.path / "config.json").write_text("{}")
    with pytest.raises(FileExistsError, match="incomplete"):
        convert.build(model)


def test_disk_space_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(convert.shutil, "disk_usage", lambda _p: Usage(0, 0, 5 * 1024**3))
    convert.check_disk_space(tmp_path / "a" / "b", 1 * 1024**3)
    with pytest.raises(OSError, match="Not enough disk space"):
        convert.check_disk_space(tmp_path / "a" / "b", 4 * 1024**3)


# ── smoke plumbing ──


def test_every_runtime_model_has_a_smoke_check() -> None:
    covered = set(smoke.CHECKS)
    assert set(config.RUNTIME_HUB_MODELS) <= covered
    assert set(config.LOCAL_MODELS) <= covered
    assert "mps-threads" in covered  # PLAN.md §2.2 M0 check


def test_run_isolated_parses_the_result_line(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        seen.update(cmd=cmd, env=kwargs["env"])
        line = json.dumps(
            {
                "key": "vad",
                "ok": True,
                "seconds": 1.0,
                "peak_footprint_gb": 0.5,
                "details": {},
                "error": "",
            }
        )
        return types.SimpleNamespace(stdout=f"noise\n{line}\n", stderr="", returncode=0)

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    result = smoke.run_isolated("vad")
    assert result.ok
    assert seen["cmd"][1:] == ["-m", "granit.models.smoke", "vad"]
    assert seen["env"]["HF_HUB_OFFLINE"] == "1"


def test_run_isolated_reports_crashes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        return types.SimpleNamespace(stdout="", stderr="Traceback\nMemoryError", returncode=1)

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    result = smoke.run_isolated("llm")
    assert not result.ok
    assert "exit 1" in result.error
    assert "MemoryError" in result.error


def test_run_in_process_catches_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing() -> dict[str, Any]:
        smoke.expect(False, "wrong answer")
        return {}

    monkeypatch.setitem(smoke.CHECKS, "fake", failing)
    result = smoke.run_in_process("fake")
    assert not result.ok
    assert result.error == "SmokeFailure: wrong answer"
    assert "❌" in result.summary()
