Feature Admission and Feature Filtering
=======================================

Provides admission strategies and filtering strategies for HashTable IDs.

Feature Admission
-----------------

.. currentmodule:: recis.nn.hashtable_hook

.. autoclass:: AdmitHook

Feature Filtering
-----------------

.. currentmodule:: recis.nn.hashtable_hook

.. autoclass:: FilterHook

Usage Examples
--------------

Admission: Feature Non-admission (Read-only Mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**HashTable**

.. code-block:: python

   from recis.nn import HashTable
   from recis.nn.hashtable_hook import AdmitHook
   ht = HashTable(
       embedding_shape=[8],
   )
   ro_hook = AdmitHook("ReadOnly")
   # Lookup embedding table in read-only mode, non-existent IDs directly return zero embeddings
   emb_r = ht(ids, admit_hook=ro_hook)

**DynamicEmbedding**

.. code-block:: python

   from recis.nn import DynamicEmbedding, EmbeddingOption
   from recis.nn.initializers import TruncNormalInitializer
   from recis.nn.hashtable_hook import AdmitHook
   
   # Configure embedding options
   emb_opt = EmbeddingOption(
       embedding_dim=64,
       shared_name="user_embedding",
       combiner="sum",
       initializer=TruncNormalInitializer(std=0.01),
       admit_hook=AdmitHook("ReadOnly"),
   )
   
   # Create dynamic embedding
   embedding = DynamicEmbedding(emb_opt)
   
   # Lookup embedding table in read-only mode, non-existent IDs directly return zero embeddings
   ids = torch.LongTensor([1, 2, 3, 4])
   emb_output = embedding(ids)

**EmbeddingEngine**

.. code-block:: python

   from recis.nn import EmbeddingEngine
   from recis.nn.hashtable_hook import AdmitHook
   
   # Configure multiple embeddings
   user_emb_opt = EmbeddingOption(
       embedding_dim=64,
       shared_name="user_emb",
       combiner="sum",
       admit_hook=AdmitHook("ReadOnly"),
   )
   id_emb_opt = EmbeddingOption(
       embedding_dim=64,
       shared_name="id_emb",
       combiner="sum"
   )
   
   # Create embedding engine
   embedding_engine = EmbeddingEngine(
       {"user_emb": user_emb_opt, "item_emb": item_emb_opt}
   )
   # Forward propagation
   samples = {
       "user_emb": user_ids,
       "item_emb": item_ids
   }
   # user_emb looks up embedding in read-only mode, item_emb looks up embedding in normal mode
   outputs = embedding_engine(samples)

Filtering: Filter Out IDs That Don't Appear for Fixed Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**EmbeddingEngine**

.. code-block:: python

   from recis.nn import EmbeddingEngine
   from recis.nn.hashtable_hook import FilterHook
   from recis.hooks.filter_hook import HashTableFilterHook
   
   # Configure multiple embeddings
   user_emb_opt = EmbeddingOption(
       embedding_dim=64,
       shared_name="user_emb",
       combiner="sum",
       # Add filtering strategy for user_emb: filter out IDs that don't appear for 10 steps
       filter_hook=FilterHook("GlobalStepFilter", {"filter_step": 10}),
   )
   id_emb_opt = EmbeddingOption(
       embedding_dim=64,
       shared_name="id_emb",
       combiner="sum"
   )
   
   # Create embedding engine
   embedding_engine = EmbeddingEngine(
       {"user_emb": user_emb_opt, "item_emb": item_emb_opt}
   )
   # Forward propagation
   samples = {
       "user_emb": user_ids,
       "item_emb": item_ids
   }
   # user_emb looks up embedding in read-only mode, item_emb looks up embedding in normal mode
   # Check for filterable IDs every 2 steps
   hook = HashTableFilterHook(2)
   for i in range(100):
       outputs = embedding_engine(samples)
       hook.after_step(None, i)