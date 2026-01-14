"""
Performance monitoring utility for diagnostics.

Provides step-by-step time and memory tracking to identify bottlenecks.
"""

import os
import sys
import time
import json
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Try to import psutil, fall back to resource module
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

try:
    import tracemalloc
    HAS_TRACEMALLOC = True
except ImportError:
    HAS_TRACEMALLOC = False


@dataclass
class StepData:
    """Data for a single performance step."""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_sec: float = 0.0
    start_rss_mb: float = 0.0
    end_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    start_vms_mb: float = 0.0
    end_vms_mb: float = 0.0
    thread_count: int = 0
    status: str = "pending"
    error: Optional[str] = None


@dataclass
class PerfReport:
    """Full performance report."""
    timestamp: str = ""
    python_version: str = ""
    platform: str = ""
    total_ram_mb: float = 0.0
    available_ram_mb: float = 0.0
    threading_env: Dict[str, str] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    total_duration_sec: float = 0.0
    peak_rss_mb: float = 0.0
    exit_code: Optional[int] = None
    exit_reason: Optional[str] = None


class PerfMonitor:
    """
    Performance monitor for tracking step-by-step execution.

    Usage:
        monitor = PerfMonitor()
        monitor.start_step("build_state")
        # ... do work ...
        monitor.end_step("build_state")
        monitor.save("runs/hash/perf.json")
    """

    def __init__(self, enabled: bool = True, trace_exceptions: bool = False):
        self.enabled = enabled
        self.trace_exceptions = trace_exceptions
        self._steps: Dict[str, StepData] = {}
        self._step_order: List[str] = []
        self._start_time = time.time()
        self._peak_rss = 0.0
        self._process = None

        if HAS_PSUTIL:
            self._process = psutil.Process()

        if HAS_TRACEMALLOC and enabled:
            tracemalloc.start()

    def _get_memory_info(self) -> tuple:
        """Get current RSS and VMS in MB."""
        rss_mb, vms_mb = 0.0, 0.0

        if HAS_PSUTIL and self._process:
            try:
                mem = self._process.memory_info()
                rss_mb = mem.rss / (1024 * 1024)
                vms_mb = mem.vms / (1024 * 1024)
            except Exception:
                pass
        elif HAS_RESOURCE:
            try:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                # maxrss is in KB on Linux
                rss_mb = usage.ru_maxrss / 1024
            except Exception:
                pass

        return rss_mb, vms_mb

    def _get_thread_count(self) -> int:
        """Get current thread count."""
        return threading.active_count()

    def snapshot(self, name: str) -> Dict[str, Any]:
        """Take a memory/time snapshot without starting a step."""
        if not self.enabled:
            return {}

        rss_mb, vms_mb = self._get_memory_info()
        elapsed = time.time() - self._start_time
        threads = self._get_thread_count()

        snapshot_data = {
            "name": name,
            "elapsed_sec": elapsed,
            "rss_mb": rss_mb,
            "vms_mb": vms_mb,
            "thread_count": threads,
            "timestamp": datetime.now().isoformat(),
        }

        # Update peak
        if rss_mb > self._peak_rss:
            self._peak_rss = rss_mb

        # Print to stdout
        print(f"[PERF] {name}: {elapsed:.2f}s, RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB, threads={threads}")

        return snapshot_data

    def start_step(self, name: str):
        """Start timing a step."""
        if not self.enabled:
            return

        rss_mb, vms_mb = self._get_memory_info()

        step = StepData(
            name=name,
            start_time=time.time(),
            start_rss_mb=rss_mb,
            start_vms_mb=vms_mb,
            thread_count=self._get_thread_count(),
            status="running",
        )

        self._steps[name] = step
        if name not in self._step_order:
            self._step_order.append(name)

        print(f"[PERF] START {name}: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB")

    def end_step(self, name: str, error: Optional[str] = None):
        """End timing a step."""
        if not self.enabled:
            return

        if name not in self._steps:
            return

        step = self._steps[name]
        step.end_time = time.time()
        step.duration_sec = step.end_time - step.start_time

        rss_mb, vms_mb = self._get_memory_info()
        step.end_rss_mb = rss_mb
        step.end_vms_mb = vms_mb
        step.peak_rss_mb = max(step.start_rss_mb, rss_mb)
        step.thread_count = self._get_thread_count()

        if error:
            step.status = "error"
            step.error = error
        else:
            step.status = "completed"

        # Update global peak
        if rss_mb > self._peak_rss:
            self._peak_rss = rss_mb

        delta_rss = step.end_rss_mb - step.start_rss_mb
        print(f"[PERF] END {name}: {step.duration_sec:.2f}s, RSS={rss_mb:.1f}MB (Δ{delta_rss:+.1f}MB)")

    def get_report(self) -> PerfReport:
        """Generate full performance report."""
        report = PerfReport(
            timestamp=datetime.now().isoformat(),
            python_version=sys.version,
            platform=sys.platform,
            threading_env={
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "not set"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "not set"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "not set"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "not set"),
            },
            total_duration_sec=time.time() - self._start_time,
            peak_rss_mb=self._peak_rss,
        )

        # Get system memory info
        if HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                report.total_ram_mb = mem.total / (1024 * 1024)
                report.available_ram_mb = mem.available / (1024 * 1024)
            except Exception:
                pass

        # Add steps in order
        for name in self._step_order:
            if name in self._steps:
                step = self._steps[name]
                report.steps.append(asdict(step))

        return report

    def save(self, filepath: str):
        """Save report to JSON file."""
        if not self.enabled:
            return

        report = self.get_report()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(asdict(report), f, indent=2)

        print(f"[PERF] Report saved to {filepath}")

    def print_summary(self):
        """Print a summary table to stdout."""
        if not self.enabled:
            return

        print("\n" + "=" * 70)
        print("PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"{'Step':<30} {'Duration':<12} {'RSS Start':<12} {'RSS End':<12}")
        print("-" * 70)

        for name in self._step_order:
            if name in self._steps:
                step = self._steps[name]
                status = "✓" if step.status == "completed" else "✗"
                print(f"{status} {name:<28} {step.duration_sec:>10.2f}s {step.start_rss_mb:>10.1f}MB {step.end_rss_mb:>10.1f}MB")

        print("-" * 70)
        print(f"Total: {time.time() - self._start_time:.2f}s, Peak RSS: {self._peak_rss:.1f}MB")
        print("=" * 70 + "\n")


# Global monitor instance
_global_monitor: Optional[PerfMonitor] = None


def get_monitor() -> Optional[PerfMonitor]:
    """Get the global monitor instance."""
    return _global_monitor


def set_monitor(monitor: PerfMonitor):
    """Set the global monitor instance."""
    global _global_monitor
    _global_monitor = monitor


def start_step(name: str):
    """Start a step using the global monitor."""
    if _global_monitor:
        _global_monitor.start_step(name)


def end_step(name: str, error: Optional[str] = None):
    """End a step using the global monitor."""
    if _global_monitor:
        _global_monitor.end_step(name, error)


def snapshot(name: str) -> Dict[str, Any]:
    """Take a snapshot using the global monitor."""
    if _global_monitor:
        return _global_monitor.snapshot(name)
    return {}
