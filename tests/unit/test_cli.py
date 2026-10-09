from __future__ import annotations

import argparse
import subprocess
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


def _pgrep_finds(monkeypatch: pytest.MonkeyPatch, line: str) -> None:
    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        assert cmd[:2] == ["pgrep", "-fl"]
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{line}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_ask_refusal_names_the_ingest_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from granit.models import download, server

    monkeypatch.setattr(download, "local_snapshot", lambda _model: tmp_path)
    monkeypatch.setattr(server.LLMServer, "healthy", lambda _self: False)
    _pgrep_finds(monkeypatch, "4321 python -m granit.ingest.worker --data lib")

    with (
        pytest.raises(
            SystemExit, match=r"ingest is running .*\(process 4321 python -m granit\.ingest"
        ),
        cli._llm_server(),
    ):
        pass


def test_verify_refusal_names_the_other_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    _pgrep_finds(monkeypatch, "999 python -m mlx_lm.server " + "x" * 200)

    with pytest.raises(SystemExit, match=r"another phase is running") as refusal:
        cli._verify(argparse.Namespace())
    message = str(refusal.value)
    assert "(process 999 python -m mlx_lm.server" in message
    assert message.endswith("x)") and "x" * 121 not in message  # the line is cut to 120 characters
