"""Process memory on macOS: physical footprint, including Metal (GPU) buffers on unified memory.

RSS misses most MLX / MPS allocations; ``phys_footprint`` is what Activity Monitor shows as "Memory" and what
counts against the ~21 GB GPU working-set limit. Read with ``proc_pid_rusage`` (no sudo needed for our own
processes). The MLX / MPS helpers import their runtimes lazily (PLAN.md §4.2).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from dataclasses import dataclass

RUSAGE_INFO_V4 = 4
GB = 1e9


class _RusageInfoV4(ctypes.Structure):
    # <sys/resource.h> struct rusage_info_v4
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        *[
            (name, ctypes.c_uint64)
            for name in (
                "ri_user_time",
                "ri_system_time",
                "ri_pkg_idle_wkups",
                "ri_interrupt_wkups",
                "ri_pageins",
                "ri_wired_size",
                "ri_resident_size",
                "ri_phys_footprint",
                "ri_proc_start_abstime",
                "ri_proc_exit_abstime",
                "ri_child_user_time",
                "ri_child_system_time",
                "ri_child_pkg_idle_wkups",
                "ri_child_interrupt_wkups",
                "ri_child_pageins",
                "ri_child_elapsed_abstime",
                "ri_diskio_bytesread",
                "ri_diskio_byteswritten",
                "ri_cpu_time_qos_default",
                "ri_cpu_time_qos_maintenance",
                "ri_cpu_time_qos_background",
                "ri_cpu_time_qos_utility",
                "ri_cpu_time_qos_legacy",
                "ri_cpu_time_qos_user_initiated",
                "ri_cpu_time_qos_user_interactive",
                "ri_billed_system_time",
                "ri_serviced_system_time",
                "ri_logical_writes",
                "ri_lifetime_max_phys_footprint",
                "ri_instructions",
                "ri_cycles",
                "ri_billed_energy",
                "ri_serviced_energy",
                "ri_interval_max_phys_footprint",
                "ri_runnable_time",
            )
        ],
    ]


@dataclass(frozen=True)
class Footprint:
    current_gb: float
    peak_gb: float  # lifetime maximum for the process
    resident_gb: float


def footprint(pid: int | None = None) -> Footprint:
    """Physical footprint of ``pid`` (default: this process). Raises OSError if it can't be read."""
    if sys.platform != "darwin":
        raise OSError("phys_footprint is only available on macOS")
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib", use_errno=True)
    info = _RusageInfoV4()
    result = libc.proc_pid_rusage(
        ctypes.c_int(pid or os.getpid()), ctypes.c_int(RUSAGE_INFO_V4), ctypes.byref(info)
    )
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"proc_pid_rusage({pid}) failed: {os.strerror(errno)}")
    return Footprint(
        current_gb=info.ri_phys_footprint / GB,
        peak_gb=info.ri_lifetime_max_phys_footprint / GB,
        resident_gb=info.ri_resident_size / GB,
    )


def mlx_peak_gb() -> float:
    """Peak MLX allocation in this process (a subset of the footprint)."""
    import mlx.core as mx

    return mx.get_peak_memory() / GB


def mps_allocated_gb() -> float:
    """Memory the PyTorch MPS driver holds for this process."""
    import torch

    return torch.mps.driver_allocated_memory() / GB


def gpu_working_set_limit_gb() -> float:
    """Metal's recommended maximum working set: the ~21 GB budget every phase must fit in (§3.3)."""
    import mlx.core as mx

    return int(mx.device_info()["max_recommended_working_set_size"]) / GB
