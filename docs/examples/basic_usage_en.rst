Basic Usage
===========

This chapter introduces the basic usage methods of RecIS to help beginners get started quickly.

Environment Setup
-----------------

Before starting, make sure you have correctly installed recis and the data extension column-io:

.. code-block:: python

   import torch
   import recis
   import column_io

Basic Concepts
--------------

**FeatureEngine**
  Feature processing engine that supports complex feature transformations

**HashTable**
  Core component for sparse parameter storage

**DynamicEmbedding**
  Dynamically expandable embedding table that supports automatic management of sparse parameters

**EmbeddingOption**
  Embedding table configuration options

**EmbeddingEngine**
  Embedding table management engine that manages multiple DynamicEmbeddings and provides optimization strategies such as sparse merging

**SparseOptimizer**
  Optimizer designed specifically for sparse parameters

**Trainer**
  Training manager

Data Processing
---------------

Data Conversion Tools
~~~~~~~~~~~~~~~~~~~~~

**Convert CSV files to ORC format data**

.. code-block:: python

   from recis.nn import DynamicEmbedding, EmbeddingOption
   
Data Reading Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    from recis.io.orc_dataset import OrcDataset

    worker_idx = int(os.environ.get("RANK", 0))
    worker_num = int(os.environ.get("WORLD_SIZE", 1))
    dataset = OrcDataset(
        1024, # batch size
        worker_idx=worker_idx,
        worker_num=worker_num,
        read_threads_num=2, # number of data reading threads
        prefetch=1, # number of prefetched data
        is_compressed=False,
        drop_remainder=True, # drop data that doesn't fill a batch
        transform_fn=[lambda x: x[0]],
        dtype=torch.float32,
        device="cuda", # dataset data results output directly to cuda
        save_interval=None,
    )
    data_paths = ["./data_dir/"]
    for path in data_paths:
        dataset.add_path(path)
    # Read fixed-length features
    dataset.fixedlen_feature("label", [0.0])
    # Read variable-length features
    dataset.varlen_feature("user_id")
    dataset.varlen_feature("item_id")

    # Build data reading
    iter = iter(dataset)
    data = next(iter)
    

Feature Engineering
-------------------

**Basic Feature Processing**

.. code-block:: python

   from recis.features import FeatureEngine
   from recis.features.feature import Feature
   from recis.features.op import SelectField, Hash, Bucketize
   
   # Define feature processing pipeline
   features = [
       # User ID hash
       Feature(
           name="user_id",
           ops=[
               SelectField("user_id"),
               Hash(bucket_size=100000)
           ]
       ),
       
       # Item ID hash
       Feature(
           name="item_id",
           ops=[
               SelectField("item_id"),
               Hash(bucket_size=50000)
           ]
       ),
       
       # Age bucketing
       Feature(
           name="age_bucket",
           ops=[
               SelectField("age"),
               Bucketize(boundaries=[18, 25, 35, 45, 55, 65])
           ]
       )
   ]
   
   # Create feature engine
   feature_engine = FeatureEngine(features)
   
   # Process data
   input_data = {
       'user_id': torch.LongTensor([[1], [2], [3]]),
       'item_id': torch.LongTensor([[101], [102], [103]]),
       'age': torch.FloatTensor([[25], [35], [45]])
   }
   
   processed_data = feature_engine(input_data)
   
   print("Original data:", input_data)
   print("Processed data:", processed_data)

Sparse Embedding Tables
------------------------

Building Embedding Tables
~~~~~~~~~~~~~~~~~~~~~~~~~

**Create Your First Embedding**

.. code-block:: python

   from recis.nn import DynamicEmbedding, EmbeddingOption
   
   # Configure embedding options
   emb_opt = EmbeddingOption(
       embedding_dim=64,
       shared_name="my_embedding",
       combiner="sum"
   )
   
   # Create dynamic embedding
   embedding = DynamicEmbedding(emb_opt)
   
   # Use embedding
   ids = torch.LongTensor([[1], [2], [3], [100], [1000]])
   emb_output = embedding(ids)
   
   print(f"Input IDs: {ids}")
   print(f"Embedding output shape: {emb_output.shape}")
   print(f"Embedding output: {emb_output}")

**Use EmbeddingEngine to Manage and Optimize Embedding Tables**

.. code-block:: python

    from recis.nn import EmbeddingEngine, EmbeddingOption
    
    # Configure embedding options
    user_emb_opt = EmbeddingOption(
        embedding_dim=64,
        shared_name="user_emb",
        combiner="sum"
    )
    id_emb_opt = EmbeddingOption(
        embedding_dim=64,
        shared_name="id_emb",
        combiner="sum"
    )
    
    # Create dynamic embedding
    embedding = EmbeddingEngine(
        {"user_emb": user_emb_opt, "item_emb": id_emb_opt}
    )
    
    # Use embedding
    user_ids = torch.LongTensor([[1], [2], [3], [100], [1000]])
    item_ids = torch.LongTensor([[11], [22], [33], [111], [1111]])
    emb_output = embedding({"user_emb": user_ids, "item_emb": item_ids})
    
    print(f"Embedding output: {emb_output}")

Build Sparse Parameter Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from recis.optim import SparseAdamW
   from recis.nn.modules.hashtable import filter_out_sparse_param
   
   # Create a simple model
   class SimpleModel(torch.nn.Module):
       def __init__(self):
           super().__init__()
           emb_opt = EmbeddingOption(embedding_dim=32)
           self.embedding = DynamicEmbedding(emb_opt)
           self.linear = torch.nn.Linear(32, 1)
       
       def forward(self, ids):
           emb = self.embedding(ids)
           return self.linear(emb)
   
   model = SimpleModel()
   
   # Separate sparse and dense parameters
   sparse_params = filter_out_sparse_param(model)
   
   print("Sparse parameters:", list(sparse_params.keys()))
   
   # Create optimizers
   sparse_optimizer = SparseAdamW(sparse_params, lr=0.001)
   dense_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

Training with Trainer
---------------------

Simple Training
~~~~~~~~~~~~~~~

.. code-block:: python

    # Build model
    # model = ...
    # Build data
    # dataset = ...
    # Define optimizers
    # sparse_params = filter_out_sparse_param(model)