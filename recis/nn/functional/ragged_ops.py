from typing import List, Optional

import torch

from recis.utils.logger import Logger


logger = Logger(__name__)


def dense_to_ragged(tensor: torch.Tensor):
    """Convert dense tensor to ragged tensor format.

    This function converts a dense tensor to ragged format by removing
    trailing zeros (padding) and creating offset arrays that define the
    boundaries of variable-length sequences. This is useful for memory
    optimization and efficient processing of sparse sequential data.

    Args:
        tensor (torch.Tensor): Input dense tensor to be converted to ragged format.
            Typically contains padded sequences where trailing zeros represent padding.

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor]]: A tuple containing:
            - values: Flattened tensor with padding removed
            - offsets: List of offset tensors defining sequence boundaries

    Example:
        >>> import torch
        >>> from recis.nn.functional.ragged_ops import dense_to_ragged
        >>> # Dense tensor with padding (zeros)
        >>> dense = torch.tensor(
        ...     [
        ...         [1, 2, 0],  # Sequence of length 2
        ...         [3, 4, 5],  # Sequence of length 3
        ...         [6, 0, 0],  # Sequence of length 1
        ...     ]
        ... )
        >>> values, offsets = dense_to_ragged(dense)
        >>> print(values)  # tensor([1, 2, 3, 4, 5, 6])
        >>> print(offsets)  # [tensor([0, 2, 5, 6])]

    Note:
        - Trailing zeros in each sequence are treated as padding and removed
        - The function assumes zero values at the end of sequences are padding
        - Resulting ragged tensor uses less memory for sparse data
        - Offset arrays enable reconstruction of original sequence boundaries
    """
    return torch.ops.recis.dense_to_ragged(tensor)


def ragged_to_sparse(values, row_splits: List[torch.Tensor]):
    """Convert ragged tensor to sparse tensor format.

    This function converts ragged tensor representation (values + offsets) to
    PyTorch's sparse COO (coordinate) tensor format. This conversion is useful
    for operations that benefit from sparse tensor optimizations or when
    interfacing with libraries that expect sparse tensors.

    Args:
        values (torch.Tensor): Flattened tensor containing all non-zero values
            from the ragged tensor.
        row_splits (List[torch.Tensor]): Offset tensor(s) defining
            the boundaries of each sequence. Can be a single tensor or list of
            tensors for multi-dimensional ragged tensors. Must have dtype int32.

    Returns:
        torch.sparse.Tensor: Sparse COO tensor representation of the ragged tensor.
            The tensor is coalesced (duplicate indices are summed).

    Raises:
        AssertionError: If row_splits tensors don't have dtype int32.
        RuntimeError: If the resulting sparse tensor has invalid dimensions.

    Example:
        >>> import torch
        >>> from recis.nn.functional.ragged_ops import ragged_to_sparse
        >>> # Ragged tensor representation
        >>> values = torch.tensor([1, 2, 3, 4, 5, 6])
        >>> row_splits = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
        >>> # Convert to sparse tensor
        >>> sparse_tensor = ragged_to_sparse(values, [row_splits])
        >>> print(sparse_tensor.indices())  # Coordinate indices
        >>> print(sparse_tensor.values())  # Non-zero values
        >>> print(sparse_tensor.size())  # Tensor dimensions
        >>> # Multi-dimensional ragged tensor
        >>> row_splits_list = [
        ...     torch.tensor([0, 2, 5, 6], dtype=torch.int32),
        ...     torch.tensor([0, 1, 3, 6, 8, 10, 11], dtype=torch.int32),
        ... ]
        >>> sparse_multi = ragged_to_sparse(values, row_splits_list)

    Note:
        - The function handles both single and multi-dimensional ragged tensors
        - Resulting sparse tensor is automatically coalesced
        - Empty sequences are properly handled in the conversion
        - GPU acceleration is supported for large-scale conversions
    """
    sparse_tensor = torch.ops.recis.ragged_to_sparse(values, row_splits)
    if not all(list(sparse_tensor.size())):
        raise RuntimeError(f"Wrong sparse tensor, got shape: {sparse_tensor.size()}")
    return sparse_tensor


def fused_ragged_cutoff_3D(
    values: List[torch.Tensor],
    offsets: List[torch.Tensor],
    keep_lengths: torch.Tensor,
    drop_sides: Optional[torch.Tensor] = None,
    pad_sides: Optional[torch.Tensor] = None,
):
    """Perform fused cutoff operations on multiple 3D ragged tensors.

    This function provides a high-level interface for the fused 3D ragged cutoff
    operation, which processes multiple 3D ragged tensors simultaneously by
    applying length constraints (cutting off excess elements or padding short
    sequences) to achieve uniform sequence lengths.

    Args:
        values (List[torch.Tensor]): List of value tensors for each 3D ragged tensor.
        offsets (List[torch.Tensor]): List of offset tensor pairs [outer_offsets, inner_offsets]
            for each 3D ragged tensor.
        keep_lengths (torch.Tensor): Target lengths for each sequence after cutoff.
        drop_sides (torch.Tensor, optional): Sides to drop from when sequences are too long.
            Defaults to None (drop from right).
        pad_sides (torch.Tensor, optional): Sides to pad when sequences are too short.
            Defaults to None (pad on right).

    Returns:
        Tuple[List[torch.Tensor], List[List[torch.Tensor]]]: A tuple containing:
            - values: List of processed value tensors
            - offsets: List of processed offset tensor pairs

    Example:
        >>> import torch
        >>> from recis.nn.functional.ragged_ops import fused_ragged_cutoff_3D
        >>> # Sample 3D ragged tensor data
        >>> values = [torch.tensor([1, 2, 3, 4, 5, 6])]
        >>> outer_offsets = torch.tensor([0, 2, 4, 6])
        >>> inner_offsets = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        >>> offsets = [[outer_offsets, inner_offsets]]
        >>> # Cutoff parameters
        >>> keep_lengths = torch.tensor([[2]], dtype=torch.int32)
        >>> # Apply cutoff
        >>> new_values, new_offsets = fused_ragged_cutoff_3D(
        ...     values, offsets, keep_lengths
        ... )

    Note:
        - This is a wrapper around the internal _fused_ragged_cutoff_3D function
        - Handles the conversion between different offset representations
        - Useful for batch processing of multiple ragged tensors
        - Supports flexible dropping and padding strategies
    """
    inner_offsets = [o[1] for o in offsets]
    outer_offsets = [o[0] for o in offsets]
    values, outer_f, inner_f = _fused_ragged_cutoff_3D(
        values, outer_offsets, inner_offsets, keep_lengths, drop_sides, pad_sides
    )
    out_off = [[outer_f[i], inner_f[i]] for i in range(len(outer_f))]
    return values, out_off


def _fused_ragged_cutoff_3D(
    values: List[torch.Tensor],
    outer_offsets: List[torch.Tensor],
    inner_offsets: List[torch.Tensor],
    keep_lengths: torch.Tensor,
    drop_sides: Optional[torch.Tensor] = None,
    pad_sides: Optional[torch.Tensor] = None,
):
    """Fused cutoff operations for multiple 3D ragged feature tensors.

    This function processes a list of 3D ragged tensors, performing a "cutoff"
    operation on the top-level sequences (rows) of each. For every tensor, it
    allows you to specify a target length (keep_lengths) for each top-level
    sequence. If a sequence is longer than its target length, inner tensors
    are dropped; if it's shorter, it is padded with empty sequences to meet
    the target length.

    The 3D ragged tensors are represented by three components:
    - values: A 1D tensor of all concatenated elements
    - outer_offsets: Offsets defining the boundaries of the top-level sequences
    - inner_offsets: Offsets defining the boundaries of the inner tensors

    The fused approach allows the operator to handle a list of tensors
    efficiently, which is common in scenarios involving multiple features
    that need to be processed in parallel.

    Args:
        values (List[torch.Tensor]): List of 1D tensors representing the
            concatenated elements of the innermost dimension. All tensors
            should be on the same device.
        outer_offsets (List[torch.Tensor]): List of 1D tensors of int32 or int64 dtype.
            Each tensor defines the top-level sequence boundaries with shape
            (num_sequences + 1,).
        inner_offsets (List[torch.Tensor]): List of 1D tensors of int32 or int64 dtype.
            Each tensor defines the boundaries of the inner tensors with length
            (num_total_inner_tensors + 1,).
        keep_lengths (torch.Tensor): 2D tensor of int32 or int64 dtype with shape
            (num_tensors, batch_size). Each element specifies the target number
            of inner tensors for a specific top-level sequence.
        drop_sides (torch.Tensor, optional): 2D tensor of bool dtype with shape
            (num_tensors, batch_size). True indicates dropping from the left (start);
            False indicates dropping from the right (end). Defaults to None (drop right).
        pad_sides (torch.Tensor, optional): 2D tensor of bool dtype with shape
            (num_tensors, batch_size). True indicates padding at the left (start);
            False indicates padding at the right (end). Defaults to None (pad right).

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
            A tuple containing the results of the cutoff operation:
            - List of new values tensors after cutoff
            - List of new outer_offsets tensors
            - List of new inner_offsets tensors

    Raises:
        ValueError: If input lists are inconsistent in length, or if tensor dtypes
            or dimensions are incorrect.
        TypeError: If an element in any of the input lists is not a torch.Tensor.

    Example:
        >>> import torch
        >>> from recis.nn.functional.ragged_ops import _fused_ragged_cutoff_3D
        >>> # Original 3D ragged tensor 'a':
        >>> # [
        >>> #     [ [10], [11, 12, 13], [] ],           # Sequence 0, length 3
        >>> #     [ [14, 15, 16, 17] ],                 # Sequence 1, length 1
        >>> #     [ [], [18, 19] ]                      # Sequence 2, length 2
        >>> # ]
        >>> values_a = torch.tensor(
        ...     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=torch.int64
        ... )
        >>> outer_offsets_a = torch.tensor([0, 3, 4, 6], dtype=torch.int32)
        >>> inner_offsets_a = torch.tensor([0, 1, 4, 4, 8, 8, 10], dtype=torch.int32)
        >>> # Cutoff parameters: keep_len = 2, drop left (True), pad left (True)
        >>> keep_lengths = torch.tensor([[2, 2, 2]], dtype=torch.int32)
        >>> drop_sides = torch.tensor([[True, True, True]], dtype=torch.bool)
        >>> pad_sides = torch.tensor([[True, True, True]], dtype=torch.bool)
        >>> # Apply cutoff
        >>> new_values, new_outer, new_inner = _fused_ragged_cutoff_3D(
        ...     [values_a],
        ...     [outer_offsets_a],
        ...     [inner_offsets_a],
        ...     keep_lengths,
        ...     drop_sides,
        ...     pad_sides,
        ... )
        >>> # Expected results:
        >>> # Sequence 0 (len 3 > 2): drop from left -> [ [11, 12, 13], [] ]
        >>> # Sequence 1 (len 1 < 2): pad left with empty -> [ [], [14, 15, 16, 17] ]
        >>> # Sequence 2 (len 2 == 2): no change -> [ [], [18, 19] ]

    Note:
        - Default behavior: drop right, pad right (when drop_sides/pad_sides are None)
        - Empty lists on different sides are not automatically dropped
        - Left-side dropping logic does not discard right-side empty lists
        - This behavior differs from sparse_cutoff operations
        - All input tensors should be on the same device for optimal performance
        - The function uses custom CUDA kernels for efficient GPU processing
    """
    # Default: drop right, pad right
    if drop_sides is None:
        drop_sides = torch.zeros(
            len(values), dtype=torch.bool, device=keep_lengths.device
        )
    else:
        if torch.any(drop_sides):
            logger.warning(
                "The empty lists on the right and left sides are not automatically dropped. "
                "That is, the left-side dropping logic does not discard the right-side empty "
                "lists — this behavior is different from that of sparse_cutoff."
            )

    if pad_sides is None:
        pad_sides = torch.zeros(
            len(values), dtype=torch.bool, device=keep_lengths.device
        )

    else:
        if torch.any(pad_sides):
            logger.warning(
                "The empty lists on the right and left sides are not automatically dropped. That is, the left-side padding logic does not discard the right-side empty lists — this behavior is different from that of sparse_cutoff."
            )

    if len(outer_offsets) != len(values) or len(outer_offsets) != len(inner_offsets):
        raise ValueError(
            f"Length of 'offsets' list ({len(outer_offsets)}) must match "
            f"length of 'values' list ({len(values)}) and 'inner_offsets' ({len(inner_offsets)}): This is critical "
            f"for ensuring each value tensor has its corresponding offsets."
        )
    if drop_sides.dtype != torch.bool:
        raise ValueError(
            f"'drop_sides' must be of type torch.bool, but got {drop_sides.dtype}."
        )
    if pad_sides.dtype != torch.bool:
        raise ValueError(
            f"'pad_sides' must be of type torch.bool, but got {pad_sides.dtype}."
        )
    return torch.ops.recis.fused_ragged_cutoff_3D(
        values, outer_offsets, inner_offsets, keep_lengths, drop_sides, pad_sides
    )


def fused_ragged_cutoff_2D(
    values: List[torch.Tensor],
    offsets: List[torch.Tensor],
    keep_lengths: torch.Tensor,
    drop_sides: torch.Tensor,
    pad_sides: torch.Tensor,
):
    """
    Args:
        values (List[torch.Tensor]): A list of 1D `torch.Tensor`s. Each tensor
                                     represents the concatenated values of a 2D
                                     ragged feature. All tensors within the list
                                     should be on the same device.
        offsets (List[torch.Tensor]): A list of 1D `torch.Tensor`s. Each tensor
                                      represents the offsets for the corresponding
                                      `values` tensor, defining its row boundaries.
                                      Each offset tensor should have a shape of
                                      `(batch_size + 1,)`. All tensors within this
                                      list must be of torch.int32 or torch.int64
                                      dtype and on the same device as `values`.
        keep_lengths (torch.Tensor): A 1D `torch.Tensor` of torch.int32 or torch.int64
                                     Its shape should be `(num_tensors, batch_size)`.
                                     Each element specifies the target length for
                                     a specific row in a specific input tensor.
                                     Must be on the same device as `values`.
        drop_sides (torch.Tensor): A 2D `torch.Tensor` of **`torch.bool`** dtype.
                                   Its shape should be `(num_tensors, batch_size)`.
                                   A value of **`True`** indicates dropping from the
                                   **left (start)**; **`False`** indicates dropping
                                   from the **right (end)**. Must be on the same
                                   device as `values`.
        pad_sides (torch.Tensor): A 2D `torch.Tensor` of **`torch.bool`** dtype.
                                  Its shape should be `(num_tensors, batch_size)`.
                                  A value of **`True`** indicates padding at the
                                  **left (start)**; **`False`** indicates padding
                                  at the **right (end)**. Must be on the same
                                  device as `values`.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
            A tuple containing six elements:
            -   `List[torch.Tensor]`: The list of new `values` tensors after cutoff.
            -   `List[torch.Tensor]`: The list of new `offsets` tensors after cutoff.
            -   `List[torch.Tensor]`: The list of `drop_nums` tensors. Each `drop_nums`
                                     tensor is 1D, indicating how many elements were
                                     dropped from each row of the corresponding input.
            -   `List[torch.Tensor]`: The list of `pad_nums` tensors. Each `pad_nums`
                                     tensor is 1D, indicating how many elements were
                                     padded to each row of the corresponding input.
            -   `torch.Tensor`: The original `drop_sides` tensor (returned as is).
            -   `torch.Tensor`: The original `pad_sides` tensor (returned as is).

    Raises:
        ValueError: If any input list is empty or inconsistent in length,
                    if tensor dimensions/dtypes are incorrect, if tensors are
                    on different devices.
        TypeError: If an element in `values` or `offsets` is not a `torch.Tensor`.
    """
    if len(offsets) != len(values):
        raise ValueError(
            f"Length of 'offsets' list ({len(offsets)}) must match "
            f"length of 'values' list ({len(values)}). This is critical "
            f"for ensuring each value tensor has its corresponding offsets."
        )
    if drop_sides.dtype != torch.bool:
        raise ValueError(
            f"'drop_sides' must be of type torch.bool, but got {drop_sides.dtype}."
        )
    if pad_sides.dtype != torch.bool:
        raise ValueError(
            f"'pad_sides' must be of type torch.bool, but got {pad_sides.dtype}."
        )

    return torch.ops.recis.fused_ragged_cutoff(
        values, offsets, keep_lengths, drop_sides, pad_sides
    )


def ragged_to_dense(values: torch.Tensor, offsets: List[torch.Tensor], default_value):
    """
    Converts a ragged tensor (represented by values and a list of offsets) into a dense tensor.

    This operator takes a set of values from a ragged tensor and its corresponding
    multi-dimensional offsets, then reshapes them into a contiguous dense tensor.
    Any positions in the target dense tensor that are not explicitly defined by
    the ragged tensor's values will be filled with a specified `default_value`.

    **Input Ragged Tensor Format:**
    * **Values:** A 1D `torch.Tensor` containing all concatenated data elements.
    * **Offsets:** A `List[torch.Tensor]` where each `torch.Tensor` defines the
        offsets for a specific dimension of the ragged tensor. The first tensor
        in the list typically represents the batch dimension, and subsequent
        tensors represent deeper nesting levels.

    **Behavior:**
    * The operator infers the output dense shape based on the provided `offsets`.
        The outermost dimension's size is determined by the first `offsets` tensor.
        Subsequent dimensions' sizes are inferred from the maximum extent defined
        by their respective offset tensors.
    * Elements from `values` are placed into their correct positions in the
        dense tensor.
    * Positions not covered by `values` are populated with `default_value`.

    **Important Considerations:**
    * **GPU Implementation Constraint:** For GPU implementations, the number of
        dimensions of the ragged tensor (i.e., the length of the `offsets` list)
        is currently **limited to less than 5**. Exceeding this limit for GPU
        inputs may lead to errors or undefined behavior.
    * **Data Type Compatibility:** Ensure that `default_value` is compatible
        with the `dtype` of the `values` tensor to avoid unexpected type
        conversions or precision loss in the output.

    Args:
        values (torch.Tensor): A 1D `torch.Tensor` containing the data elements
                                of the ragged tensor.
        offsets (List[torch.Tensor]): A list of 1D `torch.Tensor`s, where each
                                     tensor represents the offsets for a dimension.
                                     `offsets[0]` defines the first dimension's
                                     boundaries, `offsets[1]` the second, and so on.
                                     Each offset tensor should have a shape of
                                     `(num_elements_in_previous_level + 1,)`.
        default_value: The scalar value to fill in for any missing

    Returns:
        torch.Tensor: A dense `torch.Tensor` representing the converted ragged data.
                      Its rank (number of dimensions) will be equal to
                      `len(offsets)`.

    Raises:
        RuntimeError: If `offsets` list length is 5 or more when running on GPU.
                      (This error is typically raised by the underlying C++ operator).
        Other potential errors related to invalid offset formats or value counts.
    """
    for i, offset_tensor in enumerate(offsets):
        if offset_tensor.dim() != 1:
            raise ValueError(
                f"Expected offset tensor at index {i} to be 1D, but got {offset_tensor.dim()}D."
            )
    return torch.ops.recis.ragged_to_dense(values, offsets, default_value)


def feature_cross_ragged(
    x_val: torch.Tensor,
    x_offsets: torch.Tensor,
    y_val: torch.Tensor,
    y_offsets: torch.Tensor,
    x_weight: torch.Tensor = None,
    y_weight: torch.Tensor = None,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Efficiently aggregates two weighted, ragged feature pathways, generating
    Cartesian product-style hashed features.

    This function processes two logically 2D sparse feature sets (`x` and `y`),
    represented in the `(values, offsets)` ragged tensor format. During the
    cross-pathway aggregation, it considers the weight of each feature item and
    generates new combined feature IDs along with their corresponding aggregated weights.

    **Aggregation Process Overview:**
    1.  **Data Preparation**:
        * `x_val` and `y_val` are 1D Tensors representing all concatenated feature
            IDs across rows, while `x_offsets` and `y_offsets` define the start and
            end indices for each logical row.
        * If `x_weight` or `y_weight` are not provided (i.e., `None`), the function
            automatically generates a new Tensor of ones with the same shape, dtype,
            and device as the corresponding `_val` Tensor.
        * **Crucially, `x_val` and `y_val` must be 1D Tensors.**

    2.  **Row-wise Processing with Unique Operation**:
        * The function processes `x` and `y` features row by row.
        * For each row in `x` and `y`, a "unique" operation is performed on its
            `val` (feature IDs) and corresponding `weight`. This means duplicate
            feature IDs within the same row are de-duplicated, and their associated
            weights are aggregated (e.g., summed).

    3.  **Cartesian Product-style MurmurHash Aggregation**:
        * For **each pair of matching rows** from `x` and `y` (the smaller number
            of rows between `x` and `y` determines the common limit; e.g., if `x`
            has N rows and `y` has M rows with M > N, only the first N rows are processed),
            a Cartesian product-style MurmurHash operation is executed.
        * Specifically, every unique `x_id` from an `x` row is combined with every
            unique `y_id` from the corresponding `y` row, generating a new combined
            feature ID (via MurmurHash).
        * Concurrently, the weights corresponding to the `x_id` and `y_id` are
            multiplied to form the new aggregated weight for the combined feature.
            These newly generated combined feature IDs and their aggregated weights
            are stored in `out_val_tensor` and `out_weight_tensor`.

    4.  **Output Format**:
        * The final aggregated results (`out_val_tensor`, `out_weight_tensor`)
            are still in a ragged tensor format.
        * `out_offsets_tensor` defines the row boundaries for both `out_val_tensor`
            and `out_weight_tensor`.
        * **Logical Output Shape**: If `x` has N rows and `y` has M rows:
            * The number of logical output rows will be `min(N, M)`.
            * The logical width of each output row is the number of unique
                features in the corresponding `x` row multiplied by the number of
                unique features in the corresponding `y` row.
        * **Empty Row Handling**: If a logical row in either the `x` or `y` input
            (after the unique operation) is empty, the resulting output `value` and
            `weight` tensors (`out_val_tensor`, `out_weight_tensor`) will **omit**
            any corresponding entries, leading to a compact representation. However,
            the `out_offsets_tensor` will **preserve** the empty segments. For example,
            an `out_offsets_tensor` like `[0, 9, 9, 13, 36]` indicates an empty
            segment for the row starting at index 9 (`[9, 9)`), correctly representing
            an empty logical row while ensuring the offsets align with the original
            batch structure.

    **Performance Note:**
    * The performance of this operator is highly optimized when the logical length
        of each row in `x_val` is **1**.
    * If the logical length of rows in `x_val` is **greater than 1**, you may
        observe a **performance degradation**. This is due to the nature of the
        underlying C++ implementation's optimization for the `x` input's sparsity
        characteristics.

    **Usage Note:**
    * x_val, y_val must be int64; x_offsets, y_offsets must be int32 *

    * **Order of Cartesian Product:**
    *The internal implementation of this operator
        **does not guarantee the order** of the elements generated by the Cartesian
        product within each logical row. The MurmurHash values and their corresponding
        weights for a given row are deterministic, but their sequence in the
        `out_val_tensor` and `out_weight_tensor` for that row might vary across
        runs or different environments. If a specific order is required, you must
        implement sorting or ordering logic after the operator's execution.

    Args:
        x_val (torch.Tensor): Values for the first feature pathway. Must be a 1D
                              Tensor of shape `(total_x_elements,)` containing
                              all `x` feature IDs.
        x_offsets (torch.Tensor): Offsets for the first feature pathway. A 1D Tensor
                                  of shape `(batch_size + 1,)` defining the start
                                  and end indices for each row within `x_val`.
        x_weight (torch.Tensor, optional): Weights for the first feature pathway.
                                           Same shape as `x_val`. If `None`, it
                                           defaults to a Tensor of all `1.0`s.
        y_val (torch.Tensor): Values for the second feature pathway. Must be a 1D
                              Tensor of shape `(total_y_elements,)` containing
                              all `y` feature IDs.
        y_offsets (torch.Tensor): Offsets for the second feature pathway. A 1D Tensor
                                  of shape `(batch_size + 1,)` defining the start
                                  and end indices for each row within `y_val`.
        y_weight (torch.Tensor, optional): Weights for the second feature pathway.
                                           Same shape as `y_val`. If `None`, it
                                           defaults to a Tensor of all `1.0`s.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing three Tensors:
            -   **out_val_tensor (torch.Tensor)**: The aggregated combined feature IDs. A 1D Tensor.
            -   **out_offsets_tensor (torch.Tensor)**: The offsets for the aggregated results. A 1D Tensor.
            -   **out_weight_tensor (torch.Tensor)**: The aggregated combined feature weights. A 1D Tensor,
                                                     same shape as `out_val_tensor`.

    Raises:
        ValueError: If `x_val` or `y_val` are not 1D Tensors, or if provided
                    `x_weight`/`y_weight` shapes do not match their corresponding
                    `_val` Tensors.

    Example:
        ```python
        # Assume x is logically [[1, 2], [3]] with weights [w_11, w_12, w_21]
        # And y is logically [[10], [20, 30]] with weights [w_101, w_201, w_202]

        # Example input values
        x_val = torch.tensor([1, 2, 3], dtype=torch.int64)
        x_offsets = torch.tensor(
            [0, 2, 3], dtype=torch.int64
        )  # Row 0: [1, 2], Row 1: [3]
        y_val = torch.tensor([10, 20, 30], dtype=torch.int64)
        y_offsets = torch.tensor(
            [0, 1, 3], dtype=torch.int64
        )  # Row 0: [10], Row 1: [20, 30]

        # Case 1: Default weights (all ones)
        print("--- Using Default Weights ---")
        out_val, out_offsets, out_weight = feature_cross_ragged(
            x_val, x_offsets, None, y_val, y_offsets, None
        )
        # Expected logical output for Row 0:
        # Combined IDs: [murmur(1,10), murmur(2,10)]
        # Combined Weights: [1.0*1.0, 1.0*1.0]
        # Expected logical output for Row 1:
        # Combined IDs: [murmur(3,20), murmur(3,30)]
        # Combined Weights: [1.0*1.0, 1.0*1.0]
        print("Output Val (default weights):", out_val)
        print("Output Offsets (default weights):", out_offsets)
        print("Output Weight (default weights):", out_weight)
        ```
    """
    if x_val.dim() != 1:
        raise ValueError(
            f"Expected x_val to be a 1D tensor, but got a {x_val.dim()}D tensor."
        )
    if y_val.dim() != 1:
        raise ValueError(
            f"Expected y_val to be a 1D tensor, but got a {y_val.dim()}D tensor."
        )
    if x_weight is None:
        x_weight = torch.ones_like(x_val, dtype=torch.float32, device=x_val.device)
    if y_weight is None:
        y_weight = torch.ones_like(y_val, dtype=torch.float32, device=x_val.device)
    if x_weight.shape != x_val.shape:
        raise ValueError(
            f"Expected x_weight to have shape {x_val.shape} (same as x_val), "
            f"but got shape {x_weight.shape}."
        )
    if y_weight.shape != y_val.shape:
        raise ValueError(
            f"Expected y_weight to have shape {y_val.shape} (same as y_val), "
            f"but got shape {y_weight.shape}."
        )
    return torch.ops.recis.feature_cross_ragged(
        x_val, x_offsets, x_weight, y_val, y_offsets, y_weight
    )


class RaggedTile(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        batch: List[int],
        seq: List[int],
        indices: torch.Tensor,
        offset: torch.Tensor,
        table: torch.Tensor,
    ):
        [out, batch_seq] = torch.ops.recis.ragged_tile(
            batch, seq, indices, offset, table
        )
        batch_max = max(batch)
        seq_min = min(seq)
        batch_info = (
            table.shape[0],
            len(batch),
            batch_max,
            seq_min,
        )  # [table_rows, batch_len, batch_max, seq_min]
        ctx.save_for_backward(indices, offset, batch_seq)
        ctx.batch_info = batch_info
        return out

    @staticmethod
    def backward(ctx, dy):
        assert dy.is_contiguous()
        indices, offset, batch_seq = ctx.saved_tensors
        batch_info = ctx.batch_info
        dx = torch.ops.recis.ragged_tile_back(
            batch_seq, batch_info, indices, offset, dy
        )
        return None, None, None, None, dx


def ragged_tile(
    batch: List[int],
    seq: List[int],
    indices: torch.Tensor,
    offset: torch.Tensor,
    table: torch.Tensor,
    check_enable: bool = False,
):
    """
    Args:
        batch (List[int]): all tensor batch. shape=[M]
        seq (List[int]): all tensor max sequence len. shape = [M]
        indices(torch.Tensor): used to restore table.
        offset (torch.Tensor): ragged tensor offset.
        table(torch.Tensor): shape = [N,dim]
        check_enable(bool): false will not check input, true will lead to low performance
    Returns:
        torch.Tensor: shape = [batch1*seq1+...+batchM*seqM, dim]
    Example:
    >>> batch = [2, 1]
    >>> seq = [3, 4]
    >>> indices = torch.tensor([0, 1, 0, 0, 1])
    >>> table = torch.tensor([[0,1],
                              [2,3]])
    >>>  offset = torch.tensor([0,2,3,4])
    >>>  out = ragged_tile(batch, seq, indices, offset, table)
    >>>  print(out)
           [[0,1], [2,3], [0,0],
            [0,1], [0,0], [0,0],
            [2,3], [0,0], [0,0], [0,0]]
    """
    check_para(batch, seq, indices, offset, table, check_enable)
    return RaggedTile.apply(batch, seq, indices, offset, table)


def check_para(
    batch: List[int],
    seq: List[int],
    indices: torch.Tensor,
    offset: torch.Tensor,
    table: torch.Tensor,
    check_enable: bool,
):
    if not check_enable:
        return
    rows, dim = table.shape
    val_max = indices.max().item()
    off_last = offset[-1].item()
    assert len(batch) == len(seq)
    assert val_max < rows
    assert off_last <= len(indices)
    assert indices.dtype in (torch.int, torch.long)
    assert offset.dtype in (torch.int, torch.long)
    assert table.dtype in (torch.float,)
    assert indices.is_cuda and offset.is_cuda and table.is_cuda
    assert table.is_contiguous()


def ragged_topk_index_cutoff(
    drop_num: torch.Tensor,
    pad_num: torch.Tensor,
    drop_side: torch.Tensor,
    pad_side: torch.Tensor,
    offset: torch.Tensor,
    topk_index: torch.Tensor,
    indicator: torch.Tensor,
):
    """
    Applies cutoff operations to a ragged tensor based on top-k indices.

    This function performs cutoff operations on a ragged tensor using the provided
    top-k indices. It calculates the value indices and updated offsets after applying
    the cutoff operations based on drop and pad parameters.

    The function uses the RECIS calc_ragged_index operator to compute the new indices
    and offsets for the ragged tensor after considering the top-k selection.

    Args:
        drop_num (torch.Tensor): A 1D tensor specifying the number of elements being
            dropped from each segment of the ragged tensor. Shape: (num_segments,).
        pad_num (torch.Tensor): A 1D tensor specifying the number of elements being
            padded for each segment of the ragged tensor. Shape: (num_segments,).
        drop_side (torch.Tensor): A scalar boolean tensor indicating the side from
            which to drop elements. True means drop from the left/start, False means
            drop from the right/end. Shape: (num_segments,).
        pad_side (torch.Tensor): A scalar boolean tensor indicating the side to which
            to pad elements. True means pad to the left/start, False means pad to
            the right/end. Shape: (num_segments,).
        offset (torch.Tensor): A 1D tensor representing the offsets that define
            the boundaries of segments in the ragged tensor. Shape: (num_segments + 1,).
        topk_index (torch.Tensor): A 2D tensor containing the top-k indices that
            determine which elements to keep after the cutoff operations.
            Shape: (bs, keep_top).
        indicator (torch.Tensor): A 1D tensor containing the original indices of the
            rows in the topk_index.
            Shape: (bs,).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - value_index (torch.Tensor): A 1D tensor containing the computed indices of values after
              applying the cutoff operations.
            - offset (torch.Tensor): The updated offsets after applying the cutoff
              operations. Shape: (num_segments + 1,).

    Example:
        ```python
        # Example usage of ragged_topk_index_cutoff
        offset = torch.tensor([0, 8, 10, 13], dtype=torch.int32)
        drop_num = torch.tensor([5, 0, 0])
        pad_num = torch.tensor([0, 0, 0])
        drop_side = torch.tensor(True)
        pad_side = torch.tensor(False)
        topk_index = torch.tensor([[1, 2], [0, 2], [0, 1]])
        indicator = torch.tensor([0, 1, 2])
        value_index, offset = ragged_topk_index_cutoff(
            drop_num, pad_num, drop_side, pad_side, offset, topk_index, indicator
        )
        # value_index: [6, 7, 8, 10, 11]
        # offset: [0, 2, 3, 5]
        ```

    Note:
        All input tensors should be on the same device.
        The function relies on the underlying RECIS calc_ragged_index operator.
    """
    value_index, offset = torch.ops.recis.calc_ragged_index(
        drop_num, pad_num, drop_side, pad_side, offset, topk_index, indicator
    )
    return value_index, offset
