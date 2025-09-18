Dynamic Embedding Tables
========================

The RecIS Dynamic Embedding Tables provides efficient and scalable sparse parameter storage and lookup capabilities, supporting real-time updates of large-scale dynamic vocabularies and distributed training, offering complete sparse feature embedding solutions for recommendation systems and other scenarios.

Core Features
-------------

**Dynamic Embedding Management**
   - **Real-time Expanding**: Support for dynamically adding new feature IDs during training without predefined vocabulary size
   - **Feature Filtering**: Provide filtering strategies to automatically remove low-frequency or expired features

**Distributed Storage Architecture**
   - **Distributed Sharding**: Row-wise partitioning supporting multi-worker parallel training
   - **Gradient Aggregation**: Support for gradient aggregation strategies by ID or by worker

**High-Performance Computing Optimization**
   - **Operator Fusion**: Batch processing and fusion optimization for multi-feature embedding lookups
   - **GPU Acceleration**: Complete CUDA operator support fully utilizing GPU parallel computing capabilities

Single Dynamic Embedding Table
-------------------------------

.. currentmodule:: recis.nn

.. autoclass:: DynamicEmbedding
    :members: __init__, forward

Embedding Configuration
-----------------------

.. autoclass:: EmbeddingOption

Embedding Engine
----------------

.. autoclass:: EmbeddingEngine
    :members: __init__, forward
