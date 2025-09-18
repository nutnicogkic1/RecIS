import torch


def bucketize(input, boundaries):
    """Bucketize input values based on provided boundaries.

    This function maps continuous input values to discrete bucket indices
    based on the provided boundary values. It's commonly used in recommendation
    systems for discretizing continuous features like ratings, prices, or
    engagement metrics.

    Args:
        input (torch.Tensor): Input tensor containing values to be bucketized.
            Can be of any shape and should contain numeric values.
        boundaries (torch.Tensor or list): Boundary values defining the bucket
            edges. Should be sorted in ascending order. Can be provided as a
            tensor or list, which will be converted to a tensor.

    Returns:
        torch.Tensor: Tensor of bucket indices with the same shape as input.
            Each value represents the bucket index (0-based) that the
            corresponding input value falls into.

    Example:
        >>> import torch
        >>> from recis.nn.functional.array_ops import bucketize
        >>> # Define input values and boundaries
        >>> values = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])
        >>> boundaries = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> # Bucketize the values
        >>> bucket_indices = bucketize(values, boundaries)
        >>> print(bucket_indices)  # tensor([0, 1, 2, 3, 4])
        >>> # Works with multi-dimensional tensors
        >>> values_2d = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
        >>> bucket_indices_2d = bucketize(values_2d, boundaries)
        >>> print(bucket_indices_2d)  # tensor([[0, 1], [2, 3]])

    Note:
        - Values less than the first boundary are assigned to bucket 0
        - Values greater than or equal to the last boundary are assigned to the last bucket
        - The boundaries tensor is automatically moved to the same device as input
        - Input tensor is flattened internally but output maintains original shape
    """
    if not isinstance(boundaries, torch.Tensor):
        boundaries = torch.tensor(boundaries)
    boundaries = boundaries.to(device=input.device, dtype=input.dtype)
    ori_shape = input.shape
    input = input.reshape([-1])
    out = torch.ops.recis.bucketize_op(input, boundaries)
    out = out.reshape(ori_shape)
    return out


def bucketize_mod(input, bucket):
    """Apply modulo operation to input tensor for hash bucket assignment.

    This function performs unsigned 64-bit modulo operation on integer input
    tensors, commonly used for distributing IDs across hash buckets in
    recommendation systems. The operation treats input values as unsigned
    integers for consistent hash distribution.

    Args:
        input (torch.Tensor): Input tensor containing int64 values to be
            processed. Must have dtype torch.int64.
        bucket (int): Bucket size for the modulo operation. Should be a
            positive integer representing the number of hash buckets.

    Returns:
        torch.Tensor: Tensor with the same shape as input, containing the
            modulo results. Each value is in the range [0, bucket-1].

    Raises:
        AssertionError: If input tensor dtype is not torch.int64.

    Example:
        >>> import torch
        >>> from recis.nn.functional.array_ops import bucketize_mod
        >>> # Apply modulo to user IDs for hash bucketing
        >>> user_ids = torch.tensor([12345, 67890, 11111, 99999], dtype=torch.int64)
        >>> bucket_size = 1000
        >>> hash_buckets = bucketize_mod(user_ids, bucket_size)
        >>> print(hash_buckets)  # tensor([345, 890, 111, 999])
        >>> # Works with multi-dimensional tensors
        >>> user_ids_2d = torch.tensor(
        ...     [[12345, 67890], [11111, 99999]], dtype=torch.int64
        ... )
        >>> hash_buckets_2d = bucketize_mod(user_ids_2d, bucket_size)
        >>> print(hash_buckets_2d.shape)  # torch.Size([2, 2])

    Note:
        - Uses unsigned 64-bit interpretation for consistent hash distribution
        - Input tensor is flattened internally but output maintains original shape
        - Commonly used for feature hashing and load balancing in distributed systems
    """
    assert input.dtype == torch.int64
    ori_shape = input.shape
    input = input.reshape([-1])
    out = torch.ops.recis.uint64_mod(input, bucket)
    out = out.reshape(ori_shape)
    return out


def multi_hash(input, muls, primes, bucket_num):
    """Apply Multi-hash function for advanced feature hashing.

    This function implements a multi-hash algorithm that applies multiple
    hash functions to the input data, which helps reduce hash collisions
    and provides better distribution for feature hashing in recommendation
    systems. The algorithm uses multiplication and prime number operations
    for hash computation.

    Args:
        input (torch.Tensor): Input tensor containing int64 values to be hashed.
            Must have dtype torch.int64.
        muls (torch.Tensor or list): Multiplication factors for hash computation.
            Should contain exactly 4 elements. Can be provided as a tensor or
            list, which will be converted to a tensor.
        primes (torch.Tensor or list): Prime numbers for hash computation.
            Should contain exactly 4 elements. Can be provided as a tensor or
            list, which will be converted to a tensor.
        bucket_num (int): Number of hash buckets for the final hash values.

    Returns:
        List[torch.Tensor]: List of 4 tensors, each with the same shape as input,
            containing the results of the 4 different hash functions. Each tensor
            contains hash values in the range [0, bucket_num-1].

    Raises:
        AssertionError: If input dtype is not torch.int64, or if muls or primes
            don't contain exactly 4 elements.

    Example:
        >>> import torch
        >>> from recis.nn.functional.array_ops import multi_hash
        >>> # Define input IDs and hash parameters
        >>> item_ids = torch.tensor([1001, 2002, 3003, 4004], dtype=torch.int64)
        >>> muls = [31, 37, 41, 43]  # Multiplication factors
        >>> primes = [1009, 1013, 1019, 1021]  # Prime numbers
        >>> bucket_size = 1000
        >>> # Apply multi-hash
        >>> hash_results = multi_hash(item_ids, muls, primes, bucket_size)
        >>> print(len(hash_results))  # 4
        >>> print(hash_results[0].shape)  # torch.Size([4])
        >>> # Each hash function produces different results
        >>> for i, hash_vals in enumerate(hash_results):
        ...     print(f"Hash function {i}: {hash_vals}")

    Note:
        - Requires exactly 4 multiplication factors and 4 prime numbers
        - Input tensor is flattened internally but output maintains original shape
        - Multiple hash functions help reduce collision probability
        - Commonly used for feature crossing and dimensionality reduction
        - All parameters are automatically moved to the same device as input
    """
    assert input.dtype == torch.int64
    if not isinstance(muls, torch.Tensor):
        muls = torch.tensor(muls)
    muls = muls.to(device=input.device)
    if not isinstance(primes, torch.Tensor):
        primes = torch.tensor(primes)
    primes = primes.to(device=input.device)
    assert muls.numel() == 4
    assert primes.numel() == 4
    ori_shape = input.shape
    input = input.reshape([-1])
    outs = torch.ops.recis.multi_hash(input, muls, primes, bucket_num)
    outs = [out.reshape(ori_shape) for out in outs]
    return outs
