from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


@dataclass
class RaggedPadInfo:
    """Information about padding operations applied to ragged tensors.

    This dataclass stores metadata about padding and dropping operations
    that have been applied to ragged tensors, enabling proper reconstruction
    and understanding of the tensor's transformation history.

    Attributes:
        drop_nums (List[torch.Tensor], optional): Number of elements dropped
            from each dimension. Defaults to None.
        pad_nums (List[torch.Tensor], optional): Number of elements padded
            to each dimension. Defaults to None.
        drop_sides (torch.Tensor, optional): Sides from which elements were
            dropped (e.g., 'left', 'right'). Defaults to None.
        pad_sides (torch.Tensor, optional): Sides to which elements were
            padded (e.g., 'left', 'right'). Defaults to None.

    Example:
        >>> pad_info = RaggedPadInfo(
        ...     drop_nums=[torch.tensor([1, 0, 2])],
        ...     pad_nums=[torch.tensor([0, 1, 0])],
        ...     drop_sides=torch.tensor([0, 1, 0]),  # 0=left, 1=right
        ...     pad_sides=torch.tensor([1, 0, 1]),
        ... )
    """

    drop_nums: List[torch.Tensor] = None
    pad_nums: List[torch.Tensor] = None
    drop_sides: torch.Tensor = None
    pad_sides: torch.Tensor = None


class RaggedTensor:
    """Multi-dimensional tensor with variable-length dimensions.

    RaggedTensor represents tensors where one or more dimensions can have
    variable lengths. This is particularly useful for handling sequences
    of different lengths in batch processing scenarios common in
    recommendation systems and natural language processing.

    The tensor is stored in a flattened format with offset arrays that
    indicate the boundaries of each sequence in each dimension.

    Args:
        values (torch.Tensor): Flattened tensor containing all values.
        offsets (List[torch.Tensor]): List of offset tensors defining
            boundaries for each ragged dimension. Can also be a single
            tensor which will be converted to a list.
        weight (torch.Tensor, optional): Optional weight tensor with same
            shape as values. Defaults to None.
        dense_shape (tuple, optional): Shape of the equivalent dense tensor.
            If None, will be inferred from the data. Defaults to None.

    Attributes:
        _values (torch.Tensor): Internal storage for tensor values.
        _offsets (List[torch.Tensor]): Internal storage for offset arrays.
        _weight (torch.Tensor): Internal storage for weights.
        _dense_shape (tuple): Shape information for the tensor.
        _pad_info (RaggedPadInfo): Padding operation metadata.

    Example:
        >>> import torch
        >>> from recis.ragged.tensor import RaggedTensor
        >>> # Create ragged tensor manually
        >>> values = torch.tensor([1, 2, 3, 4, 5, 6])
        >>> offsets = [torch.tensor([0, 2, 5, 6])]  # 3 sequences: [1,2], [3,4,5], [6]
        >>> ragged = RaggedTensor(values, offsets)
        >>> print(ragged.shape)  # (3, 3) - 3 sequences, max length 3
        >>> # Create from dense tensor
        >>> dense = torch.tensor([[1, 2, 0], [3, 4, 5], [6, 0, 0]])
        >>> ragged = RaggedTensor.from_dense(dense)
        >>> # Convert to different formats
        >>> dense_reconstructed = ragged.to_dense()
        >>> sparse = ragged.to_sparse()
    """

    def __init__(
        self,
        values: torch.Tensor,
        offsets: List[torch.Tensor],
        weight: Optional[torch.Tensor] = None,
        dense_shape: Optional[tuple] = None,
    ):
        """Initialize a RaggedTensor.

        Args:
            values (torch.Tensor): Flattened tensor containing all values.
            offsets (List[torch.Tensor]): Offset tensors defining boundaries.
            weight (torch.Tensor, optional): Optional weight tensor. Defaults to None.
            dense_shape (tuple, optional): Dense tensor shape. Defaults to None.
        """
        self._values = values
        if isinstance(offsets, torch.Tensor):
            offsets = [offsets]
        self._offsets = offsets
        self._weight = weight
        self._dense_shape = dense_shape
        if self._dense_shape is None:
            self._dense_shape = (-1,) * (len(offsets) + 1)
            self._dense_shape = self.real_shape()
        self._dense_shape = tuple(self._dense_shape)
        self._pad_info = None

    def set_dim_by_rank(self, dim, rank):
        """Set the size of a specific dimension by rank.

        Args:
            dim (int): The dimension size to set.
            rank (int): The rank (dimension index) to modify.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> ragged.set_dim_by_rank(10, 1)  # Set dimension 1 to size 10
        """
        dense_shape = list(self._dense_shape)
        dense_shape[rank] = dim
        self._dense_shape = tuple(dense_shape)

    @property
    def dim(self):
        """Get the number of dimensions.

        Returns:
            int: Number of dimensions in the tensor.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged.dim)  # 2 for 2D ragged tensor
        """
        return len(self._dense_shape)

    @property
    def max_length(self):
        """Get the maximum length in the last dimension.

        Returns:
            int: Maximum sequence length in the last dimension.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged.max_length)  # Maximum sequence length
        """
        return self.get_shape(-1)

    @property
    def pad_info(self):
        """Get padding information.

        Returns:
            RaggedPadInfo: Padding operation metadata, or None if no padding applied.
        """
        return self._pad_info

    def set_pad_info(self, pad_info: RaggedPadInfo):
        """Set padding information for the tensor.

        Args:
            pad_info (RaggedPadInfo): Padding operation metadata to store.

        Example:
            >>> pad_info = RaggedPadInfo(drop_nums=[torch.tensor([1, 0])])
            >>> ragged.set_pad_info(pad_info)
        """
        self._pad_info = pad_info

    def real_shape(self, start=0, end=None):
        """Get the real shape of the tensor within a dimension range.

        Computes the actual shape by resolving any -1 placeholders with
        the real dimension sizes.

        Args:
            start (int, optional): Starting dimension index. Defaults to 0.
            end (int, optional): Ending dimension index. If None, uses all
                dimensions from start. Defaults to None.

        Returns:
            tuple: Real shape of the tensor in the specified range.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged.real_shape())  # (3, 3) - full shape
            >>> print(ragged.real_shape(0, 1))  # (3,) - first dimension only
        """
        if end is None:
            end = len(self._dense_shape)
        if end < 0:
            end = len(self._dense_shape) + end
        real_shape = []
        for i in range(start, end):
            s = self._dense_shape[i]
            if s == -1:
                s = self.get_shape(i)
            real_shape.append(s)
        return tuple(real_shape)

    def get_shape(self, dim):
        """Get the size of a specific dimension.

        Args:
            dim (int): Dimension index to query.

        Returns:
            int: Size of the specified dimension.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged.get_shape(0))  # Number of sequences
            >>> print(ragged.get_shape(1))  # Maximum sequence length
        """
        if self._dense_shape[dim] != -1:
            return self._dense_shape[dim]
        elif dim == 0:
            return self._offsets[0].numel() - 1
        elif self._offsets[dim - 1].numel() < 2:
            return 0
        else:
            return (
                (self._offsets[dim - 1][1:] - self._offsets[dim - 1][:-1]).max().item()
            )

    def to_dense(self, default_value: float = 0.0):
        """Convert ragged tensor to dense tensor format.

        Args:
            default_value (float, optional): Value to use for padding shorter
                sequences. Defaults to 0.0.

        Returns:
            torch.Tensor: Dense tensor with padding applied to make all
                sequences the same length.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> dense = ragged.to_dense(default_value=-1.0)
            >>> print(dense)  # Dense tensor with -1.0 padding
        """
        return torch.ops.recis.ragged_to_dense(
            self._values, self._offsets, default_value
        )

    @staticmethod
    def from_numpy(values: np.ndarray, offsets: List[np.ndarray]):
        """Create RaggedTensor from NumPy arrays.

        Args:
            values (np.ndarray): NumPy array containing flattened values.
            offsets (List[np.ndarray]): List of NumPy arrays containing offsets.
                Can also be a single array which will be converted to a list.

        Returns:
            RaggedTensor: New RaggedTensor instance created from NumPy data.

        Example:
            >>> import numpy as np
            >>> values = np.array([1, 2, 3, 4, 5])
            >>> offsets = [np.array([0, 2, 5])]
            >>> ragged = RaggedTensor.from_numpy(values, offsets)
        """
        values = torch.from_numpy(values)
        if isinstance(offsets, np.ndarray):
            offsets = [offsets]
        offsets = [torch.from_numpy(offset) for offset in offsets]
        return RaggedTensor(values, offsets)

    @staticmethod
    def from_dense(dense: torch.Tensor):
        """Create RaggedTensor from dense tensor.

        Automatically detects padding (zero values) and creates an efficient
        ragged representation by removing trailing zeros.

        Args:
            dense (torch.Tensor): Dense tensor to convert. Zero values at the
                end of sequences are treated as padding.

        Returns:
            RaggedTensor: New RaggedTensor instance with padding removed.

        Example:
            >>> dense = torch.tensor([[1, 2, 0], [3, 4, 5], [6, 0, 0]])
            >>> ragged = RaggedTensor.from_dense(dense)
            >>> print(ragged.values())  # [1, 2, 3, 4, 5, 6]
        """
        values, offsets = torch.ops.recis.dense_to_ragged(dense)
        return RaggedTensor(values, offsets, dense_shape=dense.shape)

    def to_sparse(self):
        """Convert ragged tensor to sparse tensor format.

        Returns:
            torch.sparse.Tensor: Sparse tensor representation where non-zero
                values are stored with their indices.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> sparse = ragged.to_sparse()
            >>> print(sparse.indices())  # Indices of non-zero values
            >>> print(sparse.values())  # Non-zero values
        """
        return torch.ops.recis.ragged_to_sparse(self._values, self._offsets)

    @property
    def device(self):
        """Get the device of the tensor.

        Returns:
            torch.device: Device where the tensor is stored.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged.device)  # cuda:0 or cpu
        """
        return self._values.device

    @property
    def shape(self):
        """Get the shape of the tensor.

        Returns:
            tuple: Shape tuple indicating dimensions of the tensor.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged.shape)  # (batch_size, max_length)
        """
        return self.real_shape()

    def size(self):
        """Get the size (shape) of the tensor.

        Returns:
            tuple: Shape tuple indicating dimensions of the tensor.

        Note:
            This method is provided for compatibility with PyTorch tensors.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged.size())  # Same as ragged.shape
        """
        return self.real_shape()

    def values(self):
        """Get the flattened values tensor.

        Returns:
            torch.Tensor: Flattened tensor containing all values.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> vals = ragged.values()
            >>> print(vals)  # [1, 2, 3, 4, 5, 6, ...]
        """
        return self._values

    def set_values(self, values: torch.Tensor):
        """Set new values for the tensor.

        Args:
            values (torch.Tensor): New flattened values tensor. Must have
                the same number of elements as the current values.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> new_values = torch.tensor([10, 20, 30, 40, 50, 60])
            >>> ragged.set_values(new_values)
        """
        self._values = values

    def offsets(self):
        """Get the offset tensors.

        Returns:
            List[torch.Tensor]: List of offset tensors defining sequence boundaries.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> offs = ragged.offsets()
            >>> print(offs[0])  # [0, 2, 5, 6] - boundaries for sequences
        """
        return self._offsets

    def weight(self):
        """Get the weight tensor.

        Returns:
            torch.Tensor or None: Weight tensor if available, None otherwise.

        Example:
            >>> ragged = RaggedTensor(values, offsets, weight=weights)
            >>> w = ragged.weight()
            >>> print(w)  # Weight values corresponding to each element
        """
        return self._weight

    def set_weight(self, weight: torch.Tensor):
        """Set weight tensor for the ragged tensor.

        Args:
            weight (torch.Tensor): Weight tensor with same shape as values.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> weights = torch.ones_like(ragged.values())
            >>> ragged.set_weight(weights)
        """
        self._weight = weight

    @property
    def dtype(self):
        """Get the data type of the tensor values.

        Returns:
            torch.dtype: Data type of the values tensor.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged.dtype)  # torch.float32, torch.int64, etc.
        """
        return self._values.dtype

    def pin_memory(self):
        """Create a copy of the tensor in pinned memory.

        Pinned memory enables faster CPU-GPU transfers.

        Returns:
            RaggedTensor: New RaggedTensor instance with tensors in pinned memory.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> pinned_ragged = ragged.pin_memory()
            >>> # Use pinned_ragged for faster GPU transfers
        """
        return RaggedTensor(
            self._values.pin_memory(),
            [offset.pin_memory() for offset in self._offsets],
            self._weight.pin_memory() if self._weight is not None else None,
            dense_shape=self._dense_shape,
        )

    def to(self, *args, **kwargs):
        """Move tensor to specified device or convert to specified dtype.

        Args:
            *args: Positional arguments passed to torch.Tensor.to().
            **kwargs: Keyword arguments passed to torch.Tensor.to().

        Returns:
            RaggedTensor: New RaggedTensor instance on the specified device/dtype.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> gpu_ragged = ragged.to("cuda")
            >>> float_ragged = ragged.to(torch.float32)
            >>> gpu_float_ragged = ragged.to("cuda", torch.float32)
        """
        return RaggedTensor(
            self._values.to(*args, **kwargs),
            [offset.to(*args, **kwargs) for offset in self._offsets],
            self._weight.to(*args, **kwargs) if self._weight is not None else None,
            dense_shape=self._dense_shape,
        )

    def cuda(self):
        """Move tensor to CUDA device.

        Returns:
            RaggedTensor: New RaggedTensor instance on CUDA device.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> gpu_ragged = ragged.cuda()
            >>> print(gpu_ragged.device)  # cuda:0
        """
        return RaggedTensor(
            self._values.cuda(),
            [offset.cuda() for offset in self._offsets],
            self._weight.cuda() if self._weight is not None else None,
            dense_shape=self._dense_shape,
        )

    def clone(self):
        """Create a shallow copy of the ragged tensor.

        The returned tensor shares the same underlying data but is a separate
        object that can be modified independently.

        Returns:
            RaggedTensor: Shallow copy of the ragged tensor.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> cloned = ragged.clone()
            >>> # cloned shares data with ragged but is a separate object
        """
        return RaggedTensor(
            self._values,
            list(self._offsets),
            self._weight if self._weight is not None else None,
            dense_shape=self._dense_shape,
        )

    def __str__(self) -> str:
        """Return string representation of the ragged tensor.

        Returns:
            str: Detailed string representation showing all tensor properties.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged)
            RaggedTensor(
              values=tensor([1, 2, 3, 4, 5, 6]),
              offsets=[tensor([0, 2, 5, 6])],
              weight=None,
              dense_shape=(3, 3),
              dtype=torch.int64,
              device=cpu
            )
        """
        s = (
            f"RaggedTensor(\n"
            f"  values={self._values},\n"
            f"  offsets={self._offsets},\n"
            f"  weight={self._weight},\n"
            f"  dense_shape={self._dense_shape},\n"
            f"  dtype={self._values.dtype},\n"
            f"  device={self._values.device}\n"
            f")"
        )
        return s
