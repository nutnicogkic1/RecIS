import json
from typing import Optional

import torch


class Loader:
    """Loads model state dictionaries from checkpoint files with parallel processing.

    This class handles loading both sparse (hashtable-based) and dense (tensor-based)
    state dictionaries from disk, applying filtering logic to the load configuration.

    Examples:
        Typical usage example for loading a checkpoint:

        >>> loader = Loader(
        ...     checkpoint_path="/path/to/checkpoint",
        ...     hashtables=sparse_state_dict,
        ...     tensors=dense_state_dict,
        ...     parallel=16,
        ... )
        >>> loader.load()

    """

    def __init__(
        self,
        checkpoint_path: str,
        hashtables: Optional[dict] = None,
        tensors: Optional[dict] = None,
        parallel: int = 16,
        filter_func=lambda x: x,
    ) -> None:
        """Initializes the Loader with configuration and target state dictionaries.

        Args:
            checkpoint_path: The directory path containing checkpoint files to load.
            hashtables: A dictionary to receive loaded sparse state data.
                If None, an empty dictionary will be created.
            tensors: A dictionary to receive loaded dense state data.
                If None, an empty dictionary will be created.
            parallel: The degree of parallelism for read operations. Defaults to 16.
            filter_func: A callable to filter load information. Defaults to identity function.
        """
        self._checkpoint_path = checkpoint_path
        self._hashtables = hashtables
        if self._hashtables is None:
            self._hashtables = {}
        self._tensors = tensors
        if self._tensors is None:
            self._tensors = {}
        self._impl = torch.classes.recis.Loader(
            self._checkpoint_path,
            parallel,
            self._hashtables,
            self._tensors,
        )
        self._filter_func = filter_func

    def load(self):
        """Executes the loading process.

        Retrieves default load information from the checkpoint, applies the filter function
        to modify the load configuration, and delegates to the internal loader implementation
        for actual I/O operations.

        The load operation involves:
        1. Retrieving default load information from the checkpoint metadata;
        2. Applying the filter function to modify the load configuration;
        3. Loading the state data into the provided hashtables and tensors dictionaries using parallel processing;

        The actual file reading and data reconstruction are handled by the torch.classes.recis.Loader class.
        """
        load_info = json.loads(self._impl.default_load_info())
        load_info = self._filter_func(load_info)
        self._impl.load(json.dumps(load_info))
