import json
from typing import Optional

from recis.utils.logger import Logger


logger = Logger(__name__)


class BaseHook:
    def __init__(self, name: str, params: Optional[dict] = None):
        self._name = name
        self._params = params if params is not None else {}

    @property
    def name(self) -> str:
        """Get the registered name of the hook policy.

        Returns:
            str: The name of the hook policy as registered in the system.
                This name is used to identify and instantiate the appropriate
                policy implementation in the HashtableHookFactory.
        """
        return self._name

    @property
    def params(self) -> dict:
        """Get the configuration parameters for the hook policy.

        Returns:
            dict: A dictionary containing the configuration parameters for
                the hook policy. The specific parameters depend on the policy
                implementation and requirements.
        """
        return self._params

    def __str__(self):
        """Return JSON string representation of the hook configuration.

        This method provides a standardized string representation of the hook
        that includes both the policy name and parameters in JSON format.
        This is useful for logging, debugging, and serialization purposes.

        Returns:
            str: JSON string containing the hook's name and parameters,
                with keys sorted for consistent output.
        """
        info = {"name": self._name, "params": self._params}
        return json.dumps(info, sort_keys=True)


class AdmitHook(BaseHook):
    """Feature admission hook for controlling HashTable feature acceptance.

    AdmitHook implements feature admission policies that control whether new
    features (IDs) are allowed to be added to HashTable embeddings. This is
    useful for implementing read-only modes, feature freezing, or custom
    admission criteria.

    The most common use case is the "ReadOnly" policy, which prevents new
    features from being added to the embedding table and returns zero embeddings
    for unknown IDs instead of creating new entries.

    Example:
        Read-only HashTable usage:

    .. code-block:: python

        from recis.nn import HashTable
        from recis.nn.hashtable_hook import AdmitHook

        # Create HashTable
        ht = HashTable(embedding_shape=[64])

        # Create read-only admission hook
        ro_hook = AdmitHook("ReadOnly")

        # Lookup with admission control
        # Known IDs return their embeddings, unknown IDs return zeros
        embeddings = ht(ids, admit_hook=ro_hook)


        Integration with DynamicEmbedding:

    .. code-block:: python

        from recis.nn import DynamicEmbedding, EmbeddingOption
        from recis.nn.hashtable_hook import AdmitHook

        # Configure embedding with admission hook
        emb_opt = EmbeddingOption(
            embedding_dim=64,
            shared_name="user_embedding",
            combiner="sum",
            admit_hook=AdmitHook("ReadOnly"),
        )

        # Create embedding with read-only policy
        embedding = DynamicEmbedding(emb_opt)

        # Use in inference mode (no new embeddings created)
        ids = torch.LongTensor([1, 2, 3, 4])
        emb_output = embedding(ids)


        Multi-embedding setup with selective admission:

    .. code-block:: python

        from recis.nn import EmbeddingEngine, EmbeddingOption
        from recis.nn.hashtable_hook import AdmitHook

        # Configure different admission policies
        user_emb_opt = EmbeddingOption(
            embedding_dim=64,
            shared_name="user_emb",
            admit_hook=AdmitHook("ReadOnly"),  # Read-only for users
        )

        item_emb_opt = EmbeddingOption(
            embedding_dim=64,
            shared_name="item_emb",
            # No admission hook = normal mode (new items allowed)
        )

        # Create embedding engine
        embedding_engine = EmbeddingEngine(
            {"user_emb": user_emb_opt, "item_emb": item_emb_opt}
        )

        # Mixed mode: user embeddings read-only, item embeddings normal
        samples = {"user_emb": user_ids, "item_emb": item_ids}
        outputs = embedding_engine(samples)

    """

    @property
    def type(self) -> str:
        """Get the hook type identifier.

        Returns:
            str: Always returns "admit" to identify this as an admission hook.
                This type identifier is used by the system to distinguish
                between different hook categories.
        """
        return "admit"

    def __str__(self):
        """Return JSON string representation of the admission hook.

        This method provides a standardized string representation of the
        admission hook configuration in JSON format, including the policy
        name and parameters.

        Returns:
            str: JSON string containing the hook's name and parameters,
                with keys sorted for consistent output.
        """
        info = {"name": self._name, "params": self._params}
        return json.dumps(info, sort_keys=True)


class FilterHook(BaseHook):
    """Feature filtering hook for implementing HashTable cleanup strategies.

    FilterHook implements feature filtering policies that automatically remove
    unused or outdated features from HashTable embeddings. This helps manage
    memory usage and maintain embedding table quality by removing features
    that are no longer relevant.

    The most common policy is "GlobalStepFilter", which removes features that
    haven't been accessed for a specified number of training steps. This is
    particularly useful in online learning scenarios where feature relevance
    changes over time.

    Example:
        Basic filtering with step-based cleanup:

    .. code-block:: python

        from recis.nn import EmbeddingEngine, EmbeddingOption
        from recis.nn.hashtable_hook import FilterHook
        from recis.hooks.filter_hook import HashTableFilterHook

        # Configure embedding with filtering policy
        user_emb_opt = EmbeddingOption(
            embedding_dim=64,
            shared_name="user_emb",
            combiner="sum",
            # Remove IDs not seen for 10 steps
            filter_hook=FilterHook("GlobalStepFilter", {"filter_step": 20}),
        )

        # Create embedding engine
        embedding_engine = EmbeddingEngine({"user_emb": user_emb_opt})

        # Setup filtering hook for periodic cleanup
        filter_hook = HashTableFilterHook(filter_interval=10)  # Check every 10 steps

        # Training loop with automatic filtering
        for step in range(100):
            outputs = embedding_engine(samples)

            # Trigger filtering check
            filter_hook.after_step(None, step)

            if step % 10 == 0:
                print(f"Step {step}: Automatic cleanup performed")


        Advanced filtering configuration:

    .. code-block:: python

        # Multiple embeddings with different filtering policies
        user_emb_opt = EmbeddingOption(
            embedding_dim=64,
            shared_name="user_emb",
            # Aggressive filtering for user features
            filter_hook=FilterHook("GlobalStepFilter", {"filter_step": 5}),
        )

        item_emb_opt = EmbeddingOption(
            embedding_dim=64,
            shared_name="item_emb",
            # Conservative filtering for item features
            filter_hook=FilterHook("GlobalStepFilter", {"filter_step": 50}),
        )

        category_emb_opt = EmbeddingOption(
            embedding_dim=32,
            shared_name="category_emb",
            # No filtering for stable category features
        )

        # Create engine with mixed filtering policies
        embedding_engine = EmbeddingEngine(
            {
                "user_emb": user_emb_opt,
                "item_emb": item_emb_opt,
                "category_emb": category_emb_opt,
            }
        )

    """

    @property
    def type(self) -> str:
        """Get the hook type identifier.

        Returns:
            str: Always returns "filter" to identify this as a filtering hook.
                This type identifier is used by the system to distinguish
                between different hook categories.
        """
        return "filter"

    def __str__(self):
        """Return JSON string representation of the filtering hook.

        This method provides a standardized string representation of the
        filtering hook configuration in JSON format, including the policy
        name and parameters.

        Returns:
            str: JSON string containing the hook's name and parameters,
                with keys sorted for consistent output.
        """
        info = {"name": self._name, "params": self._params}
        return json.dumps(info, sort_keys=True)
