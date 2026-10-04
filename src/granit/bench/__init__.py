"""M1 benchmarks (``granit bench``): speed, memory per phase and phase-switch time (PLAN.md §3.3, §5).

The orchestrator (``run.py``) never loads a model. Each phase runs in its own process (``workers.py`` or
``mlx_lm.server``), as it will in production, and memory is read as each process's physical footprint.
"""
