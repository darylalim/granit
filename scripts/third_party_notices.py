"""Print THIRD_PARTY_NOTICES.md for every installed distribution (PLAN.md §4.4).

CI runs this in the macOS `test` job, where all dependencies are installed, and the `release` job attaches it.
Usage: uv run python scripts/third_party_notices.py > THIRD_PARTY_NOTICES.md
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import distribution
from pathlib import Path

from granit.licenses import canonical_name, installed, license_texts, load_overrides

ROOT = Path(__file__).resolve().parents[1]


def render() -> str:
    packages = [
        p for p in installed(load_overrides(ROOT / "licenses_overrides.toml")) if p.name != "granit"
    ]
    out = [
        "# Third-party notices",
        "",
        "granit is licensed under Apache-2.0 (see LICENSE and NOTICE). It depends on the packages below,",
        "each under its own license. Model weights are not distributed; they are downloaded from Hugging Face.",
        "",
        "| Package | Version | License |",
        "|---|---|---|",
        *(f"| {p.name} | {p.version} | {p.license} |" for p in packages),
        "",
    ]
    for p in packages:
        out += [f"## {p.name} {p.version}", "", f"License: {p.license}", ""]
        texts = license_texts(distribution(p.name))
        if not texts:
            out += ["(No license file shipped in the distribution metadata.)", ""]
        for name, text in texts:
            # A fence longer than any backtick run inside the text keeps the license verbatim.
            fence = "`" * max([3, *(len(run) + 1 for run in re.findall(r"`+", text))])
            out += [f"<details><summary>{canonical_name(p.name)}: {name}</summary>", ""]
            out += [f"{fence}text", text.rstrip(), fence, "", "</details>", ""]
    return "\n".join(out)


if __name__ == "__main__":
    sys.stdout.write(render())
