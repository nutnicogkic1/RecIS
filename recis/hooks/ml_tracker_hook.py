from typing import Dict

import ml_tracker
import torch

from recis.hooks import Hook
from recis.utils.logger import Logger


logger = Logger(__name__)

TRACE_MAP = {}


def get_trace_map():
    """Gets the global trace map containing data to be logged.

    Returns:
        Dict: Global trace map containing key-value pairs of data to be tracked.

    Example:
        >>> add_to_ml_tracker("loss", 0.123)
        >>> add_to_ml_tracker("accuracy", 0.95)
        >>> trace_data = get_trace_map()
        >>> print(trace_data)
        {"loss": 0.123, "accuracy": 0.95}
    """
    global TRACE_MAP
    return TRACE_MAP


def clear_trace_map():
    """Clears the global trace map.

    This function is typically called after logging data to the ML tracker
    to prepare for the next batch of metrics.

    Example:
        >>> add_to_ml_tracker("loss", 0.123)
        >>> clear_trace_map()
        >>> trace_data = get_trace_map()
        >>> print(trace_data)
        {}
    """
    global TRACE_MAP
    TRACE_MAP = {}


def add_to_ml_tracker(name: str, data):
    """Adds data to the ML tracker trace map.

    This function adds metrics or other data to the global trace map that will
    be logged to the ML tracking system. Tensor data is automatically converted
    to numpy arrays for compatibility.

    Args:
        name (str): Name/key for the data being tracked.
        data: Data to be tracked. Can be torch.Tensor or any other type.
            Tensors are automatically converted to CPU numpy arrays.

    Example:
        >>> import torch
        >>> # Add scalar metrics
        >>> add_to_ml_tracker("loss", 0.123)
        >>> add_to_ml_tracker("learning_rate", 0.001)
        >>> # Add tensor data (automatically converted)
        >>> predictions = torch.tensor([0.1, 0.9, 0.3])
        >>> add_to_ml_tracker("predictions", predictions)
        >>> # The tensor will be stored as numpy array in trace map

    Note:
        Tensor data is detached from the computation graph and moved to CPU
        before conversion to numpy to ensure compatibility with the ML tracker.
    """
    global TRACE_MAP
    if isinstance(data, torch.Tensor):
        TRACE_MAP[name] = data.detach().cpu().numpy()
    else:
        TRACE_MAP[name] = data


class MLTrackerHook(Hook):
    """Hook for experiment tracking with ML tracking systems.

    The MLTrackerHook integrates with ML tracking platforms to automatically
    log training metrics, hyperparameters, and other experiment data. It
    initializes an ML tracker session and logs accumulated data after each
    training step.

    Args:
        project (str): Name of the project for experiment tracking.
        name (str): Name of the experiment run.
        config (Dict): Configuration dictionary containing hyperparameters
            and other experiment settings.
        id (optional): Unique identifier for the experiment run. If None,
            a new ID will be generated automatically.

    Attributes:
        tracker: ML tracker instance for logging experiment data.

    Example:
        >>> from recis.hooks import MLTrackerHook, add_to_ml_tracker
        >>> # Create ML tracker hook
        >>> config = {
        ...     "learning_rate": 0.001,
        ...     "batch_size": 32,
        ...     "model_type": "transformer",
        ... }
        >>> ml_hook = MLTrackerHook(
        ...     project="recommendation_model", name="experiment_v1", config=config
        ... )
        >>> trainer.add_hook(ml_hook)
        >>> # During training, add metrics to be tracked
        >>> add_to_ml_tracker("train_loss", loss.item())
        >>> add_to_ml_tracker("train_accuracy", accuracy)
        >>> # The hook will automatically log these metrics after each step

    Note:
        This hook is only available in internal environments where the
        ml_tracker library is accessible. Use add_to_ml_tracker() to add
        data that should be logged to the tracking system.
    """

    def __init__(self, project: str, name: str, config: Dict, id=None) -> None:
        super().__init__()
        self.tracker = ml_tracker.init(project=project, name=name, id=id, config=config)

    def after_step(self, metric, global_step):
        """Called after each training step to log accumulated metrics.

        This method retrieves all data from the trace map and logs it to the
        ML tracking system with the current global step number. After logging,
        the trace map is cleared to prepare for the next step.

        Args:
            metric: Training metrics from the current step (unused in this implementation).
            global_step (int): Global training step number for timestamping the logged data.

        Note:
            The method logs all data accumulated in the global trace map via
            add_to_ml_tracker() calls since the last step, then clears the map.
        """
        data = get_trace_map()
        self.tracker.log(data, global_step)
        clear_trace_map()

    def end(self):
        """Called at the end of training to finalize the ML tracker session.

        This method properly closes the ML tracker session, ensuring all
        logged data is saved and the experiment is marked as completed.
        """
        self.tracker.finish()
