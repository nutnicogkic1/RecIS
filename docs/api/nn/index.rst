Neural Network Module
=====================

RecIS's neural network module provides components optimized for sparse computation, including dynamic embedding, sparse merging, feature filtering and other core functionalities.

Components
----------

.. toctree::
   :maxdepth: 2

   hashtable
   filter
   embedding
   initializer
   functional

Component Overview
------------------

.. list-table:: Component Description
   :header-rows: 1
   :widths: 20 80

   * - Component
     - Description
   * - :doc:`hashtable`
     - Core implementation of dynamic embedding tables
   * - :doc:`filter`
     - Feature admission and feature filtering strategies
   * - :doc:`embedding`
     - High-level wrapper for dynamically expandable embedding tables, supporting distributed and sparse merging performance optimizations
   * - :doc:`initializer`
     - Initializers
   * - :doc:`functional`
     - Other functional interfaces

Best Practices
--------------

**Performance Optimization**

1. **Reasonable block_size Setting**:

   .. code-block:: python

      # Adjust block_size based on memory size
      emb_opt = EmbeddingOption(
          embedding_dim=64,
          block_size=8192,  # Smaller block_size suitable for memory-constrained environments
          shared_name="embedding"
      )

2. **Use Appropriate Initializers**:

   .. code-block:: python

      # For deep networks, use smaller initialization standard deviation
      from recis.nn.initializers import TruncNormalInitializer
      
      emb_opt = EmbeddingOption(
          embedding_dim=128,
          initializer=TruncNormalInitializer(std=0.001)
      )

3. **Enable coalesced Optimization**:

   .. code-block:: python

      # For multi-table queries, enabling coalesced can improve performance
      emb_opt = EmbeddingOption(
          embedding_dim=64,
          coalesced=True,
          shared_name="coalesced_emb"
      )

**Distributed Training**

.. code-block:: python

   import torch.distributed as dist
   
   # Initialize distributed environment
   dist.init_process_group()
   
   # Create distributed embedding
   emb_opt = EmbeddingOption(
       embedding_dim=64,
       pg=dist.group.WORLD,
       grad_reduce_by="worker"
   )
   embedding = DynamicEmbedding(emb_opt)

**Memory Management**

.. code-block:: python

   # For large-scale embedding, use half precision
   emb_opt = EmbeddingOption(
       embedding_dim=256,
       dtype=torch.float16,
       device=torch.device("cuda")
   )

Common Questions
----------------

**Q: How to handle variable-length sequence embedding?**

A: Use RaggedTensor as input:

.. code-block:: python

   from recis.ragged.tensor import RaggedTensor
   
   # Create RaggedTensor
   values = torch.LongTensor([1, 2, 3, 4, 5])
   offsets = torch.LongTensor([0, 2, 5])  # Two sequences: [1,2] and [3,4,5]
   ragged_input = RaggedTensor(values, offsets)
   
   # Use embedding
   output = embedding(ragged_input)

**Q: How to share embedding parameters?**

A: By setting the same `shared_name`:

.. code-block:: python

   # Two embeddings share parameters
   user_emb = DynamicEmbedding(EmbeddingOption(
       embedding_dim=64,
       shared_name="shared_emb"
   ))
   
   item_emb = DynamicEmbedding(EmbeddingOption(
       embedding_dim=64,
       shared_name="shared_emb"  # Same name
   ))

**Q: How to optimize embedding lookup performance?**

A: You can optimize through the following ways:

1. Enable `coalesced=True` for batch query optimization
2. Use appropriate `grad_reduce_by` strategy
3. Set correct device when computing on GPU