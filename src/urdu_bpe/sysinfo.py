"""Machine description for the report ("report the wall-clock time and the machine you used")."""

from __future__ import annotations

import os
import platform
import resource
import sys


def machine_info() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            cpu = next(
                line.split(":", 1)[1].strip() for line in fh if line.startswith("model name")
            )
    except (OSError, StopIteration):
        pass
    ram_gb = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            ram_gb = round(
                int(next(line for line in fh if line.startswith("MemTotal")).split()[1]) / 1e6, 1
            )
    except (OSError, StopIteration, ValueError):
        pass
    return {
        "cpu": cpu,
        "logical_cores": os.cpu_count(),
        "ram_gb": ram_gb,
        "os": platform.platform(),
        "python": sys.version.split()[0],
    }


def peak_rss_gb() -> float:
    """Peak resident memory of this process in GB (ru_maxrss is KiB on Linux)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
