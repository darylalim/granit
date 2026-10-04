"""Run ``mlx_lm.server`` with an MLX buffer-cache limit: ``python -m granit.models.mlx_server --mlx-cache-limit-gb 1 <server args>``.

M1 finding: MLX keeps freed buffers in a cache for reuse. As the KV cache grows during a long prefill, the
old buffers pile up there: a 30K-token request peaked at a 19.8 GB footprint with the default cache, but
16.3 GB with a 1 GB limit, at the same speed. mlx_lm.server has no option for this, hence the wrapper.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

GB = 1e9


def split_args(argv: Sequence[str]) -> tuple[float | None, list[str]]:
    """Separate our ``--mlx-cache-limit-gb`` from the arguments passed through to mlx_lm.server."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mlx-cache-limit-gb", type=float)
    known, rest = parser.parse_known_args(list(argv))
    return known.mlx_cache_limit_gb, rest


def main(argv: Sequence[str] | None = None) -> None:
    limit_gb, server_args = split_args(sys.argv[1:] if argv is None else argv)

    import mlx.core as mx

    if limit_gb is not None:
        mx.set_cache_limit(int(limit_gb * GB))

    from mlx_lm.server import main as server_main

    sys.argv = ["mlx_lm.server", *server_args]
    server_main()


if __name__ == "__main__":
    main()
