"""Diagnostics package for performance monitoring."""

from .perf import PerfMonitor, start_step, end_step, snapshot, get_monitor

__all__ = ["PerfMonitor", "start_step", "end_step", "snapshot", "get_monitor"]
