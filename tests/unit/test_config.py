"""Model licensing guard (PLAN.md §4.2): the CI counterpart of hook H3.

APPROVED is maintained here, separately from config.py, so adding or swapping a model needs a deliberate
change to this list in the same PR.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from granit import config

APPROVED = {
    "ibm-granite/granite-speech-5.0-470m-turboctc": "apache-2.0",
    "mlx-community/silero-vad-v6": "mit",
    "ibm-granite/granite-docling-258M-mlx": "apache-2.0",
    "ibm-granite/granite-vision-4.1-4b": "apache-2.0",
    "ibm-granite/granite-4.2-8b-q8-mlx": "apache-2.0",
    "ibm-granite/granite-embedding-english-r2": "apache-2.0",
    "ibm-granite/granite-embedding-reranker-english-r2": "apache-2.0",
    "ibm-granite/granite-guardian-4.1-8b": "apache-2.0",
}
PERMISSIVE_MODEL_LICENSES = {"apache-2.0", "mit"}

MODELS = list(config.HUB_MODELS.values())


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.key)
def test_model_is_approved(model: config.HubModel) -> None:
    assert model.repo_id in APPROVED, f"{model.repo_id} is not on the approved model list"
    assert model.license == APPROVED[model.repo_id]
    assert model.license in PERMISSIVE_MODEL_LICENSES


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.key)
def test_no_non_commercial_variants(model: config.HubModel) -> None:
    assert not re.search(r"-nc($|[-_/])", model.repo_id, re.IGNORECASE), model.repo_id


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.key)
def test_revision_is_pinned_to_a_commit(model: config.HubModel) -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", model.revision), f"{model.key}: pin a full commit SHA"


def test_keys_match_dict_keys() -> None:
    assert all(k == m.key for k, m in config.HUB_MODELS.items())
    assert all(k == m.key for k, m in config.LOCAL_MODELS.items())


def test_local_models_come_from_pinned_sources() -> None:
    for model in config.LOCAL_MODELS.values():
        assert model.source in config.HUB_MODELS
        assert config.HUB_MODELS[model.source].phase == "convert"
        assert model.path.is_relative_to(config.MODELS_DIR)


def test_guardian_source_is_not_a_runtime_download() -> None:
    assert "guardian-source" not in config.RUNTIME_HUB_MODELS
    assert set(config.RUNTIME_HUB_MODELS) == {
        "speech",
        "vad",
        "docling",
        "vision",
        "llm",
        "embedding",
        "reranker",
    }


def test_cached_snapshot_follows_hf_cache_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    model = config.HUB_MODELS["vad"]
    assert model.cached_snapshot() is None
    snapshot = tmp_path / "models--mlx-community--silero-vad-v6" / "snapshots" / model.revision
    snapshot.mkdir(parents=True)
    assert model.cached_snapshot() is None  # empty directory = interrupted download
    (snapshot / "config.json").write_text("{}")
    assert model.cached_snapshot() == snapshot


def test_hf_cache_resolution_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert config.hf_hub_cache() == tmp_path / "home" / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "direct"))
    assert config.hf_hub_cache() == tmp_path / "direct"


def test_local_model_is_built_needs_source_record(tmp_path: Path) -> None:
    model = config.LocalModel("x", "guardian-source", tmp_path / "m", 8, "C", 1.0)
    assert not model.is_built()
    model.path.mkdir()
    (model.path / "config.json").write_text("{}")
    assert not model.is_built()  # conversion may have died before finishing
    (model.path / config.SOURCE_RECORD).write_text("{}")
    assert model.is_built()


def test_config_is_stdlib_only() -> None:
    """Hooks import config.py with the system python3, so it must not import third-party packages."""
    source = Path(config.__file__).read_text()
    imports = re.findall(r"^(?:from|import) (\w+)", source, re.MULTILINE)
    assert set(imports) <= {"__future__", "os", "dataclasses", "pathlib"}
