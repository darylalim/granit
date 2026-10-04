from __future__ import annotations

from pathlib import Path

import pytest

from granit import cli, config


def test_help_lists_model_actions(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["models", "--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for action in ("list", "download", "convert", "smoke"):
        assert action in out


def test_unknown_model_key_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["models", "download", "gpt-5"])
    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_models_list_reports_cache_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    vad = config.HUB_MODELS["vad"]
    snapshot = tmp_path / "models--mlx-community--silero-vad-v6" / "snapshots" / vad.revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_text("x")

    assert cli.main(["models", "list"]) == 0
    rows = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()[1:]}
    assert "downloaded" in rows["vad"]
    assert "missing" in rows["llm"]
    assert set(rows) == set(config.HUB_MODELS) | set(config.LOCAL_MODELS)


def test_download_defaults_to_runtime_models(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fetched = []
    monkeypatch.setattr(
        "granit.models.download.download", lambda m: fetched.append(m.key) or Path("/x")
    )
    assert cli.main(["models", "download"]) == 0
    assert fetched == list(config.RUNTIME_HUB_MODELS)
    assert "guardian-source" not in fetched
