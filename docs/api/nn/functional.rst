Other Functional Interfaces
===========================

RecIS also provides other functional operation interfaces:

array_ops
---------

.. currentmodule:: recis.nn.functional.array_ops

.. function:: bucketize(input, boundaries)

    Data bucketing

.. function:: bucketize_mod(input, bucket)

    Data modulo

.. function:: multi_hash(input, muls, primes, bucket_num)

    Multi-hash

embedding_ops
-------------

.. module:: recis.nn.functional.embedding_ops

.. function:: ids_partition(ids, max_partition_num, world_size)

   ID partitioning function for distributed scenarios.

.. function:: ids_encode(ids_list, table_ids)
   
   ID encoding function.

.. function:: merge_offsets(offsets)
   
   Offset merging function.

.. function:: ragged_embedding_segment_reduce(emb, offsets, combiner="sum")

   Ragged Tensor format embedding aggregation.

.. function:: sparse_embedding_segment_reduce(unique_emb, weight, reverse_indices, segment_ids, num_segments, combiner)

   Sparse Tensor format embedding segment aggregation.

fused_ops
---------

.. module:: recis.nn.functional.fused_ops

.. function:: fused_bucketize_gpu(values: List[torch.Tensor], boundaries: List[torch.Tensor])

   Multi-group data bucketing operation processing

.. function:: fused_uint64_mod_gpu(values: List[torch.Tensor], mods: Union[List, torch.Tensor])

   Multi-group data modulo operation processing

.. function:: fused_ids_encode_gpu(ids_list: List[torch.Tensor], table_ids: Union[torch.Tensor, list])

   Multi-group data ID encoding operation processing

.. function:: fused_multi_hash( inputs: List[torch.Tensor], muls: List[torch.Tensor], primes: List[torch.Tensor], bucket_nums: List[torch.Tensor],)

   Multi-group data multi-hash operation processing

.. function:: fused_multi_hash( inputs: List[torch.Tensor], muls: List[torch.Tensor], primes: List[torch.Tensor], bucket_nums: List[torch.Tensor],)

   Multi-group data multi-hash operation processing

.. module:: recis.nn.functional.ragged_ops

.. function:: fused_ragged_cutoff_2D(values: List[torch.Tensor], offsets: List[torch.Tensor], keep_lengths: torch.Tensor, drop_sides: torch.Tensor, pad_sides: torch.Tensor)

   Multi-group 2D ragged tensor cutoff operation processing

.. function:: fused_ragged_cutoff_3D(values: List[torch.Tensor], offsets: List[torch.Tensor], keep_lengths: torch.Tensor, drop_sides: torch.Tensor, pad_sides: torch.Tensor)

   Multi-group 3D ragged tensor cutoff operation processing

hash_ops
--------

.. module:: recis.nn.functional.hash_ops

.. function:: farmhash(inputs, splits)

   Multi-group data farm hash operation processing

.. function:: murmurhash(inputs, splits)

   Multi-group data murmur hash operation processing
