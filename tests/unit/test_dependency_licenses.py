"""Dependency-license guard (PLAN.md §4.4): no copyleft, restricted or unknown licenses in the environment."""

from __future__ import annotations

from email.message import Message
from pathlib import Path
from typing import Any, cast

import pytest

from granit import licenses
from tests.conftest import ROOT

OVERRIDES = licenses.load_overrides(ROOT / "licenses_overrides.toml")


def test_every_installed_package_has_an_allowed_license() -> None:
    packages = licenses.installed(OVERRIDES)
    assert len(packages) > 50  # sanity: we really scanned the environment
    rejected = [
        f"{p.name} {p.version}: {p.license} ({p.reason})" for p in packages if not p.allowed
    ]
    assert not rejected, (
        "Disallowed or unknown dependency licenses (fix, or add a reviewed entry to "
        "licenses_overrides.toml):\n" + "\n".join(rejected)
    )


def test_overrides_are_still_needed_and_allowed() -> None:
    installed = {licenses.canonical_name(p.name) for p in licenses.installed({})}
    for name, entry in OVERRIDES.items():
        assert name in installed, f"override for {name} is stale: package not installed"
        assert licenses.expression_allowed(entry["license"]), (
            f"{name}: override license not allowed"
        )


def test_overrides_require_a_reason(tmp_path: Path) -> None:
    path = tmp_path / "o.toml"
    path.write_text('[pkg]\nlicense = "MIT"\n')
    with pytest.raises(ValueError, match="reason"):
        licenses.load_overrides(path)


@pytest.mark.parametrize(
    ("expression", "allowed"),
    [
        ("MIT", True),
        ("Apache-2.0 OR GPL-2.0-only", True),
        ("(MIT OR GPL-3.0-or-later) AND BSD-3-Clause", True),
        ("Apache-2.0 WITH LLVM-exception", True),
        ("MPL-2.0 AND MIT", True),
        ("GPL-3.0-only", False),
        ("LGPL-2.1-or-later", False),
        ("AGPL-3.0", False),
        ("MIT AND GPL-2.0-only", False),
        ("Elastic-2.0", False),
        ("BUSL-1.1", False),
        ("SSPL-1.0", False),
        ("CC-BY-NC-4.0", False),
        ("Proprietary", False),
        ("", False),
    ],
)
def test_expression_allowed(expression: str, allowed: bool) -> None:
    assert licenses.expression_allowed(expression) is allowed


class FakeDist:
    def __init__(self, **fields: Any) -> None:
        self.metadata = Message()
        self.metadata["Name"] = fields.pop("name", "pkg")
        for classifier in fields.pop("classifiers", []):
            self.metadata["Classifier"] = classifier
        for key, value in fields.items():
            self.metadata[key.replace("_", "-").title()] = value
        self.version = "1.0"


def detect(**fields: Any) -> tuple[str, str]:
    return licenses.detect(cast(Any, FakeDist(**fields)))


def test_detect_prefers_license_expression() -> None:
    assert detect(license_expression="BSD-3-Clause", license="whatever") == (
        "BSD-3-Clause",
        "expression",
    )


def test_detect_uses_classifiers_as_alternatives() -> None:
    expr, source = detect(
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "License :: OSI Approved :: Apache Software License",
        ]
    )
    assert (expr, source) == ("MIT OR Apache-2.0", "classifier")


def test_detect_flags_gpl_classifier() -> None:
    expr, _ = detect(
        classifiers=["License :: OSI Approved :: GNU General Public License v3 (GPLv3)"]
    )
    assert not licenses.expression_allowed(expr)


def test_detect_free_text_alias_and_fingerprint() -> None:
    assert detect(license="Apache 2.0 License") == ("Apache-2.0", "text")
    mit_text = "Copyright (c) 2020 X\n\n" + "Permission is hereby granted, free of charge, " * 5
    assert detect(license=mit_text) == ("MIT", "text")


def test_detect_unknown() -> None:
    assert detect() == ("UNKNOWN", "none")
    assert detect(license="UNKNOWN") == ("UNKNOWN", "none")


def test_decode_falls_back_to_latin1() -> None:
    assert licenses.decode("© 2026".encode()) == "© 2026"
    assert licenses.decode(b"\xa9 2026") == "© 2026"  # Latin-1 copyright sign


def test_classify_rejects_unknown() -> None:
    result = licenses.classify(cast(Any, FakeDist(name="mystery")), {})
    assert not result.allowed
    assert result.reason == "no license metadata"
