IO Module
=========

RecIS's IO module provides efficient and flexible data loading and preprocessing capabilities, supporting multiple data formats and optimized data pipelines for deep learning model training. With RecIS's IO module, you can achieve better performance without needing to combine with traditional DataLoader.

Core Features
-------------

**Data Support Support**
   - **ORC Files**: Support for Optimized Row Columnar format, suitable for large-scale offline data processing

**High-Performance Data Processing**
   - Multi-threaded parallel reading and data preprocessing
   - Configurable prefetching and buffering mechanisms
   - Direct data organization on different devices (CPU/GPU/Pin Memory)

**Flexible Feature Configuration**
   - Support for sparse features (variable-length) and dense features (fixed-length)
   - Hash feature processing with FarmHash and MurmurHash algorithms
   - RaggedTensor format for variable-length features

**Distributed Training Optimization**
   - Multi-worker data sharding
   - State saving and recovery mechanisms

.. currentmodule:: recis.io

Dataset Classes
---------------

OrcDataset
~~~~~~~~~~

.. autoclass:: OrcDataset

   :members: __init__, add_path, add_paths, varlen_feature, fixedlen_feature

Best Practices
--------------

1. **Reasonable Parameter Settings**:

   .. code-block:: python

      dataset = OrcDataset(
          batch_size=1024,
          read_threads_num=2,
          prefetch=1,           # Number of prefetch batches
          device="cuda"         # Put batch results directly on cuda
      )

2. **Using Data Preprocessing**:

   .. code-block:: python

      def transform_batch(batch):
          # Custom batch processing logic
          return processed_batch
      
      dataset = OrcDataset(
          batch_size=1024,
          transform_fn=transform_batch
      )

3. **Distributed Data Reading**:

   .. code-block:: python

      import os
      
      rank = int(os.environ.get("RANK", 0))
      world_size = int(os.environ.get("WORLD_SIZE", 1))
      
      dataset = OrcDataset(
          batch_size=1024,
          worker_idx=rank,
          worker_num=world_size
      )

Common Questions
----------------

**Q: How to handle variable-length sequences?**

A: Use `varlen_feature` to define variable-length features, RecIS will automatically process them into RaggedTensor format:

.. code-block:: python

   dataset.varlen_feature("sequence_ids")
   # Data will be processed as RaggedTensor, containing values and offsets

**Q: How to customize data preprocessing?**

A: Pass a custom processing function through the `transform_fn` parameter:

.. code-block:: python

   def custom_transform(batch):
       # Custom processing logic
       batch['processed_feature'] = process_feature(batch['raw_feature'])
       return batch
   
   dataset = OrcDataset(batch_size=1024, transform_fn=custom_transform)

**Q: How to optimize data reading performance?**

A: You can optimize from the following aspects:

1. Modify `read_threads_num` and `prefetch` parameters
2. Set reasonable `batch_size`
3. Set device='cuda' to automatically organize output results on cuda
