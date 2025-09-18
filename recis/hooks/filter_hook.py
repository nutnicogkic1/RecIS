import torch

from recis.hooks.hook import Hook
from recis.nn.modules.hashtable_hook_impl import HashtableHookFactory
from recis.utils.logger import Logger


class HashTableFilterHook(Hook):
    """Hook for automatic hash table feature filtering during training.

    This hook manages the lifecycle of features in hash tables by coordinating
    filtering operations across multiple hash table instances. It automatically
    updates step counters and triggers filtering operations at configurable
    intervals to remove stale or inactive features.

    The hook integrates with the hash table filter system to:
        - Track global training steps for each hash table filter
        - Execute filtering operations at specified intervals
        - Provide comprehensive logging of filter activities
        - Support dynamic adjustment of filtering frequency

    Args:
        filter_interval (int, optional): Number of training steps between
            filter operations. If None, filtering is disabled. Defaults to 100.

    Examples:

    Please refer to the documentation :doc:`nn/filter`

    .. code-block:: python

        # Create and configure filter hook
        filter_hook = HashTableFilterHook(filter_interval=200)

        # Training loop integration
        for epoch in range(num_epochs):
            for step, batch in enumerate(dataloader):
                # ... training logic ...

                # Hook automatically manages filtering
                filter_hook.after_step(None, global_step)

                global_step += 1
    """

    def __init__(self, filter_interval: int = 100):
        """Initialize the hash table filter hook.

        Args:
            filter_interval (int, optional): Number of training steps between
                filter operations. Must be positive. If None, filtering is
                disabled. Defaults to 100.

        Example:

        .. code-block:: python

            # Standard filtering every 100 steps
            hook = HashTableFilterHook(filter_interval=100)"""
        super().__init__()
        self.filter_interval = filter_interval
        self.ht_filters = HashtableHookFactory().get_filters()
        self.logger = Logger("HashTableFilterHook")
        self.last_filter_step = 0
        self.logger.info(
            f"HashTableFilterHook {self.ht_filters}, filter_interval {self.filter_interval}"
        )

    def reset_filter_interval(self, interval: int = 100):
        """Reset the filtering interval to a new value.

        This method allows dynamic adjustment of the filtering frequency
        during training, which can be useful for adaptive memory management
        or performance optimization strategies.

        Args:
            interval (int, optional): New filtering interval in training steps.
                Must be positive. Defaults to 100.
        """
        self.filter_interval = interval

    def after_step(self, _, gstep):
        """Execute filter management operations after each training step.

        This method is called after each training step to:
        1. Update step counters for all registered hash table filters
        2. Determine if filtering should be executed based on the interval
        3. Trigger filtering operations when the interval is reached
        4. Update the last filter step tracking

        Args:
            _ (Any): Unused parameter (typically model or trainer instance).
            gstep (Union[int, torch.Tensor]): Current global training step.
                Can be either an integer or a tensor containing the step value.

        Note:
            This method is typically called automatically by the training
            framework's hook system. Manual calls should ensure proper
            step sequencing to maintain filtering accuracy.
        """
        if self.last_filter_step == 0:
            self.last_filter_step = (
                gstep.item() if isinstance(gstep, torch.Tensor) else gstep
            )
        exec_filter = (
            self.filter_interval is not None
            and gstep - self.last_filter_step >= self.filter_interval
        )
        for hooks in self.ht_filters.values():
            for ft in hooks.values():
                ft.update_step()
                if exec_filter:
                    ft.do_filter()
        if exec_filter:
            self.last_filter_step = (
                gstep.item() if isinstance(gstep, torch.Tensor) else gstep
            )
