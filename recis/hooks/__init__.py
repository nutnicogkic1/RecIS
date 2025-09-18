from recis.info import is_internal_enabled

from .filter_hook import HashTableFilterHook
from .hook import Hook
from .logger_hook import LoggerHook
from .profiler_hook import ProfilerHook


__all__ = ["Hook", "LoggerHook", "ProfilerHook", "HashTableFilterHook"]

if is_internal_enabled():
    from .ml_tracker_hook import (
        MLTrackerHook as MLTrackerHook,
        add_to_ml_tracker as add_to_ml_tracker,
    )
    from .trace_to_odps_hook import (
        TraceToOdpsHook as TraceToOdpsHook,
        add_to_trace as add_to_trace,
    )

    __all__.extend(
        ["MLTrackerHook", "add_to_ml_tracker", "TraceToOdpsHook", "add_to_trace"]
    )
