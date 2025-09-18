import time
from typing import Dict

from recis.hooks.hook import Hook
from recis.utils.logger import Logger


class LoggerHook(Hook):
    """Hook for logging training metrics and progress.

    The LoggerHook logs training metrics at regular intervals and provides
    performance statistics including queries per second (QPS). This hook is
    automatically added by the Trainer, so manual addition is typically not required.

    Args:
        log_step (int): Logging interval in steps. Defaults to 10.

    .. note:
        The LoggerHook is automatically added by the Trainer, so manual addition is typically not required.

    Example:
        >>> from recis.hooks import LoggerHook
        >>> # Create logger hook with custom interval
        >>> from recis.framework.metrics import add_metric
        >>> add_metric("loss", 0.123)
        >>> add_metric("accuracy", 0.95)
        >>> logger_hook = LoggerHook(log_step=50)
        >>> trainer.add_hook(logger_hook)
        >>> # The hook will automatically log metrics every 50 steps
        >>> # Output format: <gstep=100> <lstep=50> <qps=12.34> <loss=0.123> <accuracy=0.95>

    Note:
        The Trainer automatically adds a LoggerHook, so manual addition is usually
        not necessary unless you need custom logging intervals or multiple loggers.
    """

    def __init__(self, log_step=10):
        self.log_step = log_step
        self.logger = Logger("Metirc")
        self.steps = 0
        self.start_time = time.time()

    def format_metrics(self, metrics: Dict):
        """Formats metrics dictionary into a readable log string.

        Converts a dictionary of metrics into a formatted string suitable for logging.
        Each metric is formatted as <key=value> for easy parsing and readability.

        Args:
            metrics (Dict): Dictionary containing metric names and values.

        Returns:
            str: Formatted string containing all metrics in <key=value> format.

        Example:
            >>> hook = LoggerHook()
            >>> metrics = {"loss": 0.123, "accuracy": 0.95}
            >>> formatted = hook.format_metrics(metrics)
            >>> print(formatted)
            " <loss=0.123> <accuracy=0.95>"
        """
        log_str = ""
        for k, v in metrics.items():
            log_str += f" <{k}={v}>"
        return log_str

    def after_step(self, metrics, global_step=0):
        """Called after each training step to potentially log metrics.

        This method is invoked after each training step. It logs metrics and
        performance statistics at the specified interval (log_step). The log
        includes global step, local step, QPS, and all provided metrics.

        Args:
            metrics (Dict): Dictionary of metrics to log (e.g., loss, accuracy).
            global_step (int): Global training step number across all epochs.
                Defaults to 0.

        Example:
            The logged output will look like:
            "<gstep=100> <lstep=50> <qps=12.34> <loss=0.123> <accuracy=0.95>"

            Where:
            - gstep: Global step number
            - lstep: Local step number (within this hook instance)
            - qps: Queries/steps per second over the last interval
            - Additional metrics as provided in the metrics dictionary
        """
        if self.steps % self.log_step == 0:
            spend_time = time.time() - self.start_time
            qps = self.log_step / spend_time
            log_str = f"<gstep={global_step}> <lstep={self.steps}> <qps={qps:.4f}>"
            log_str += self.format_metrics(metrics)
            self.logger.info(log_str)
            self.start_time = time.time()
        self.steps += 1
