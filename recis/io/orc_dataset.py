import os

import torch

from recis.io.dataset_base import DatasetBase
from recis.utils.logger import Logger


if not os.environ.get("BUILD_DOCUMENT", None) == "1":
    import column_io.dataset.dataset as column_io_dataset
    from column_io.dataset.file_sharding import OrcFileSharding


logger = Logger(__name__)


def get_dir_size(dir_name):
    """Get the size of a directory.

    Args:
        dir_name (str): The directory path to measure.

    Returns:
        int: The size of the directory. Currently returns 0 as placeholder.

    Todo:
        Implement actual file size calculation logic.
    """
    # TODO(yuhuan.zh) get file real size
    return 0


class OrcDataset(DatasetBase):
    """ORC Dataset for reading Optimized Row Columnar format files.

    This class provides functionality to read ORC files efficiently with support for
    both sparse (variable-length) and dense (fixed-length) features. It extends
    DatasetBase to provide ORC-specific optimizations including hash feature processing,
    data sharding, and batch processing.

    The OrcDataset supports distributed training by allowing multiple workers to
    process different shards of the data concurrently. It also provides flexible
    feature configuration with hash bucketing for categorical features.

    Attributes:
        hash_types (List[str]): List of hash algorithms used for features.
        hash_buckets (List[int]): List of hash bucket sizes for features.
        hash_features (List[str]): List of feature names that use hashing.

    Example:
        Creating and configuring an ORC dataset:

    .. code-block:: python

        # Initialize dataset
        dataset = OrcDataset(
            batch_size=512, worker_idx=0, worker_num=4, shuffle=True, ragged_format=True
        )

        # Add data sources
        dataset.add_paths(["/data/train/part1", "/data/train/part2"])

        # Configure sparse features with hashing
        dataset.varlen_feature("item_id", hash_type="farm", hash_bucket=1000000)
        dataset.varlen_feature("category_id", hash_type="murmur", hash_bucket=10000)

        # Configure dense features
        dataset.fixedlen_feature("price", default_value=0.0)
        dataset.fixedlen_feature("rating", default_value=3.0)

    """

    def __init__(
        self,
        batch_size,
        worker_idx=0,
        worker_num=1,
        read_threads_num=4,
        pack_threads_num=None,
        prefetch=1,
        is_compressed=False,
        drop_remainder=False,
        worker_slice_batch_num=None,
        shuffle=False,
        ragged_format=True,
        transform_fn=None,
        save_interval=100,
        dtype=torch.float32,
        device="cpu",
    ) -> None:
        """Initialize OrcDataset with configuration parameters.

        Args:
            batch_size (int): Number of samples per batch.
            worker_idx (int, optional): Index of current worker. Defaults to 0.
            worker_num (int, optional): Total number of workers. Defaults to 1.
            read_threads_num (int, optional): Number of reading threads. Defaults to 4.
            pack_threads_num (int, optional): Number of packing threads. Defaults to None.
            prefetch (int, optional): Number of batches to prefetch. Defaults to 1.
            is_compressed (bool, optional): Whether data is compressed. Defaults to False.
            drop_remainder (bool, optional): Whether to drop incomplete batches. Defaults to False.
            worker_slice_batch_num (int, optional): Number of batches per worker slice. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
            ragged_format (bool, optional): Whether to use ragged tensor format. Defaults to True.
            transform_fn (callable, optional): Data transformation function. Defaults to None.
            save_interval (int, optional): Interval for saving checkpoints. Defaults to 100.
            dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
            device (str, optional): Device for tensor operations. Defaults to "cpu".

        Raises:
            AssertionError: If is_compressed is True (not supported yet).

        Note:
            Compressed data is not currently supported for ORC datasets.
        """
        super().__init__(
            batch_size,
            worker_idx,
            worker_num,
            read_threads_num,
            pack_threads_num,
            prefetch,
            is_compressed,
            drop_remainder,
            worker_slice_batch_num,
            ragged_format,
            transform_fn,
            save_interval,
            dtype,
            device,
        )
        assert not is_compressed, "OrcDataset not support compressed data yet."
        self._shuffle = shuffle
        self._dir_sizes = []
        self._total_row_count = 0
        self.hash_types = []
        self.hash_buckets = []
        self.hash_features = []

    def add_path(self, file_path):
        """Add a single file path to the dataset.

        Args:
            file_path (str): Path to the ORC file or directory to be added.

        Example:

        .. code-block:: python

            dataset.add_path("/data/train/part_001.orc")

        """
        self._paths.append(file_path)
        self._dir_sizes.append(0)

    def add_paths(self, file_paths):
        """Add multiple file paths to the dataset.

        Args:
            file_paths (List[str]): List of paths to ORC files or directories.

        Example:

        .. code-block:: python

            dataset.add_paths(
                [
                    "/data/train/part_001.orc",
                    "/data/train/part_002.orc",
                    "/data/train/part_003.orc",
                ]
            )

        """
        for file_path in file_paths:
            self.add_path(file_path)

    def varlen_feature(self, name, hash_type=None, hash_bucket=0, trans_int8=False):
        """Configure a variable-length (sparse) feature with optional hashing.

        Variable-length features are columns that contain sequences or lists of values
        with varying lengths across samples. These features can optionally be processed
        with hash functions for dimensionality reduction and categorical encoding.

        Args:
            name (str): Name of the feature column in the ORC files.
            hash_type (str, optional): Hash algorithm to use for the feature.
                Supported values are "farm" (FarmHash) and "murmur" (MurmurHash).
                If None, no hashing is applied. Defaults to None.
            hash_bucket (int, optional): Size of the hash bucket (vocabulary size).
                Only used when hash_type is specified. Defaults to 0.
            trans_int8 (bool, optional): Whether to convert string data directly to
                int8 tensors without hashing. Only effective when hash_type is None.
                Defaults to False.

        Example:

        .. code-block:: python

            # Sparse feature with FarmHash
            dataset.varlen_feature(
                "item_sequence", hash_type="farm", hash_bucket=100000
            )

            # Sparse feature with MurmurHash
            dataset.varlen_feature(
                "category_ids", hash_type="murmur", hash_bucket=50000
            )

            # Raw sparse feature without hashing
            dataset.varlen_feature("user_tags")

            # String feature converted to int8
            dataset.varlen_feature("text_tokens", trans_int8=True)


        Raises:
            AssertionError: If hash_type is not "farm" or "murmur" when specified.

        Note:
            Hash functions are useful for handling large categorical vocabularies
            by mapping them to a fixed-size space. FarmHash generally provides
            better distribution properties, while MurmurHash is faster.
        """
        if name not in self._select_column:
            self._select_column.append(name)
            if hash_type:
                assert hash_type in [
                    "farm",
                    "murmur",
                ], "hash_type must be farm / murmur"
                self.hash_features.append(name)
                self.hash_buckets.append(hash_bucket)
                self.hash_types.append(hash_type)
            elif trans_int8:
                self.hash_features.append(name)
                self.hash_buckets.append(hash_bucket)
                self.hash_types.append("no_hash")

    def fixedlen_feature(self, name, default_value):
        """Configure a fixed-length (dense) feature.

        Fixed-length features are typically used for numerical data where each
        sample has exactly one value, such as user age, item price, or ratings.

        Args:
            name (str): Name of the feature column.
            default_value (float): Default value to use when feature is missing.

        Example:

        .. code-block:: python

            # Numerical features with default values
            dataset.fixedlen_feature("user_age", default_value=25.0)
            dataset.fixedlen_feature("item_price", default_value=0.0)
            dataset.fixedlen_feature("rating", default_value=3.0)


        Note:
            Default values are important for handling missing data gracefully
            and ensuring consistent tensor shapes across batches.
        """
        if name not in self._select_column:
            self._select_column.append(name)
        if name not in self._dense_column:
            self._dense_column.append(name)
            self._dense_default_value.append(default_value)

    def _shard_path(self, sub_id, sub_num):
        """Create data shards for distributed processing.

        This method partitions the input files across multiple workers and threads
        to enable parallel data loading. It uses OrcFileSharding to ensure
        balanced distribution of data.

        Args:
            sub_id (int): Sub-process identifier within the worker.
            sub_num (int): Total number of sub-processes per worker.

        Note:
            This is an internal method used by the dataset creation process.
            The sharding strategy ensures that each worker processes a unique
            subset of the data while maintaining load balance.
        """
        file_shard = OrcFileSharding()
        file_shard.add_paths(self._paths)
        self._shard_paths = file_shard.partition(
            self._worker_idx * sub_num + sub_id,
            self._worker_num * sub_num,
            self._read_threads_num,
            shuffle=self._shuffle,
        )

    def make_dataset_fn(self):
        """Create a dataset factory function for ORC file processing.

        This method returns a lambda function that creates a column_io Dataset
        from ORC files with the configured features and processing parameters.

        Returns:
            callable: A function that takes a file path and returns a Dataset object.

        Note:
            The returned function is used internally by the data loading pipeline
            to create dataset instances for each shard of data.
        """
        return lambda x: column_io_dataset.Dataset.from_orc_files(
            [x.decode() if isinstance(x, bytes) else x],
            self._is_compressed,
            self._batch_size,
            self._select_column,
            self.hash_features,
            self.hash_types,
            self.hash_buckets,
            self._dense_column,
            self._dense_default_value,
        )

    def get_dir_size(self):
        """Calculate and return the sizes of all configured data directories.

        This method iterates through all added paths and calculates their sizes,
        updating the internal tracking of total row count for the dataset.

        Returns:
            List[int]: List of directory sizes corresponding to each path.

        Note:
            Currently uses a placeholder implementation that returns 0 for all
            directories. The actual size calculation logic needs to be implemented.
        """
        for i, dir_name in enumerate(self._paths):
            dir_size = get_dir_size(dir_name)
            self._dir_sizes[i] = dir_size
            self._total_row_count += dir_size
        return self._dir_sizes
