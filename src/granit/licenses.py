"""Dependency license classification for the license guard and third-party notices (PLAN.md §4.4).

Reads installed distributions' metadata with ``importlib.metadata`` (offline, fast). The order of evidence is
PEP 639 ``License-Expression`` → trove classifiers → the free-text ``License`` field. Anything not clearly
permissive is rejected unless ``licenses_overrides.toml`` records a hand-reviewed exception.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from importlib.metadata import Distribution, distributions
from pathlib import Path

ALLOWED = frozenset(
    {
        "MIT",
        "MIT-0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "Apache-2.0",
        "ISC",
        "PSF-2.0",
        "Python-2.0",
        "Zlib",
        "HPND",
        "0BSD",
        "Unlicense",
        "CC0-1.0",
        # Permissive OSI licenses found in the 2026-10-04 audit: Pillow (HPND variant),
        # regex (Python 1.6 license), torch (Boost Software License; not BUSL).
        "MIT-CMU",
        "CNRI-Python",
        "BSL-1.0",
        # Weak copyleft limited to changes made to its own files; we use these unmodified (PLAN.md §4.4).
        "MPL-2.0",
    }
)
REJECTED_MARKERS = ("GPL", "AGPL", "SSPL", "BUSL", "Elastic", "CC-BY-NC", "Commons-Clause", "-NC")

# Trove classifier → SPDX id.
CLASSIFIERS = {
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: MIT No Attribution License (MIT-0)": "MIT-0",
    "License :: OSI Approved :: BSD License": "BSD-3-Clause",
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: ISC License (ISCL)": "ISC",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    "License :: OSI Approved :: zlib/libpng License": "Zlib",
    "License :: OSI Approved :: Historical Permission Notice and Disclaimer (HPND)": "HPND",
    "License :: OSI Approved :: The Unlicense (Unlicense)": "Unlicense",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication": "CC0-1.0",
}

# Free-text License field → SPDX id (matched case-insensitively against the whole short value).
TEXT_ALIASES = {
    "mit": "MIT",
    "mit license": "MIT",
    "the mit license": "MIT",
    "mit-0": "MIT-0",
    "bsd": "BSD-3-Clause",
    "bsd license": "BSD-3-Clause",
    "new bsd": "BSD-3-Clause",
    "new bsd license": "BSD-3-Clause",
    "3-clause bsd": "BSD-3-Clause",
    "3-clause bsd license": "BSD-3-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd 3-clause": "BSD-3-Clause",
    "bsd 3-clause license": "BSD-3-Clause",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd 2-clause": "BSD-2-Clause",
    "apache 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "apache 2": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache 2.0 license": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "apache software license 2.0": "Apache-2.0",
    "isc": "ISC",
    "isc license": "ISC",
    "psf": "PSF-2.0",
    "psf-2.0": "PSF-2.0",
    "python software foundation license": "PSF-2.0",
    "mpl-2.0": "MPL-2.0",
    "mpl 2.0": "MPL-2.0",
    "mozilla public license 2.0 (mpl 2.0)": "MPL-2.0",
    "zlib": "Zlib",
    "hpnd": "HPND",
    "unlicense": "Unlicense",
    "0bsd": "0BSD",
}

# Fingerprints for packages that paste the whole license text into the License field.
TEXT_FINGERPRINTS = (
    ("Permission is hereby granted, free of charge", "MIT"),
    ("Apache License", "Apache-2.0"),
    ("Redistribution and use in source and binary forms", "BSD-3-Clause"),
    ("Permission to use, copy, modify, and/or distribute this software for any", "ISC"),
    ("Mozilla Public License Version 2.0", "MPL-2.0"),
)

SHORT_TEXT = 120


@dataclass(frozen=True)
class PackageLicense:
    name: str
    version: str
    license: str  # SPDX expression, or "UNKNOWN"
    source: str  # where it came from: expression | classifier | text | override | none
    allowed: bool
    reason: str = ""


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_overrides(path: Path) -> dict[str, dict[str, str]]:
    """``[package]`` tables with ``license`` and ``reason`` (both required)."""
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text())
    overrides = {}
    for name, entry in data.items():
        if not isinstance(entry, dict) or not entry.get("license") or not entry.get("reason"):
            raise ValueError(f"{path.name}: [{name}] needs both `license` and `reason`")
        overrides[canonical_name(name)] = entry
    return overrides


def expression_allowed(expression: str) -> bool:
    """Evaluate a simple SPDX expression: ``OR`` needs one allowed side, ``AND`` / ``WITH`` need all."""
    expr = expression.strip()
    while expr.startswith("(") and expr.endswith(")") and _balanced(expr[1:-1]):
        expr = expr[1:-1].strip()
    for op, combine in ((" OR ", any), (" AND ", all)):
        parts = _split_top_level(expr, op)
        if len(parts) > 1:
            return combine(expression_allowed(p) for p in parts)
    if (
        " WITH " in expr
    ):  # e.g. "Apache-2.0 WITH LLVM-exception": the exception only adds permissions
        expr = expr.split(" WITH ", 1)[0].strip()
    return expr in ALLOWED


def _balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth < 0:
            return False
    return depth == 0


def _split_top_level(expr: str, op: str) -> list[str]:
    parts, depth, start, i = [], 0, 0, 0
    while i < len(expr):
        ch = expr[i]
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth == 0 and expr.startswith(op, i):
            parts.append(expr[start:i])
            start = i + len(op)
            i = start
            continue
        i += 1
    parts.append(expr[start:])
    return [p.strip() for p in parts]


def detect(dist: Distribution) -> tuple[str, str]:
    """Return (SPDX expression or "UNKNOWN", evidence source) for one installed distribution."""
    meta = dist.metadata
    if expression := (meta.get("License-Expression") or "").strip():
        return expression, "expression"

    found = []
    for classifier in meta.get_all("Classifier") or []:
        if classifier in CLASSIFIERS:
            found.append(CLASSIFIERS[classifier])
        elif classifier.startswith("License ::") and any(m in classifier for m in REJECTED_MARKERS):
            found.append(classifier.rsplit("::", 1)[-1].strip())
    if found:
        # Several license classifiers usually mean "choose one" (dual licensing).
        return " OR ".join(dict.fromkeys(found)), "classifier"

    text = (meta.get("License") or "").strip()
    if text and text.upper() != "UNKNOWN":
        if len(text) <= SHORT_TEXT:
            alias = TEXT_ALIASES.get(text.lower().rstrip("."))
            if alias:
                return alias, "text"
            if expression_allowed(text):
                return text, "text"
        else:
            for fingerprint, spdx in TEXT_FINGERPRINTS:
                if fingerprint in text[:2000]:
                    return spdx, "text"
    return "UNKNOWN", "none"


def classify(dist: Distribution, overrides: dict[str, dict[str, str]]) -> PackageLicense:
    name = dist.metadata["Name"]
    version = dist.version
    override = overrides.get(canonical_name(name))
    if override:
        expr = override["license"]
        return PackageLicense(
            name, version, expr, "override", expression_allowed(expr), override["reason"]
        )
    expr, source = detect(dist)
    if expr == "UNKNOWN":
        return PackageLicense(name, version, expr, source, False, "no license metadata")
    if any(
        marker.lower() in expr.lower() for marker in REJECTED_MARKERS
    ) and not expression_allowed(expr):
        return PackageLicense(name, version, expr, source, False, "copyleft / restricted license")
    allowed = expression_allowed(expr)
    return PackageLicense(
        name, version, expr, source, allowed, "" if allowed else "not on allow-list"
    )


def installed(overrides: dict[str, dict[str, str]]) -> list[PackageLicense]:
    seen: dict[str, PackageLicense] = {}
    for dist in distributions():
        name = dist.metadata["Name"]
        if not name or canonical_name(name) in seen:
            continue
        seen[canonical_name(name)] = classify(dist, overrides)
    return sorted(seen.values(), key=lambda p: canonical_name(p.name))


def license_texts(dist: Distribution) -> list[tuple[str, str]]:
    """License / notice files shipped in a distribution's metadata (PEP 639 ``licenses/`` or legacy names)."""
    texts = []
    for file in dist.files or []:
        parts = file.parts
        if not parts or not parts[0].endswith(".dist-info"):
            continue
        name = parts[-1].upper()
        if "licenses" in parts or name.startswith(
            ("LICENSE", "LICENCE", "COPYING", "NOTICE", "AUTHORS")
        ):
            content = decode(file.read_binary())
            if content.strip():
                texts.append(("/".join(parts[1:]), content))
    return texts


def decode(data: bytes) -> str:
    """Some license files are Latin-1 (e.g. a raw ``©`` byte); never fail the notices over an encoding."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")
