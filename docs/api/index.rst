API Documentation
=================

RecIS provides rich API interfaces. This section details the usage of each module.

.. toctree::
   :maxdepth: 2

   io
   features
   nn/index
   metrics
   hooks
   optim
   framework
   serialize
   utils

Core Module Overview
--------------------

.. list-table:: Core Module Description
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Description
   * - :doc:`io`
     - Data reading and preprocessing, supports local Orc file data sources
   * - :doc:`features`
     - Feature engineering module, supports feature processing and transformation
   * - :doc:`nn/index`
     - Model-related modules, including dynamic embedding and special operators provided by recis
   * - :doc:`metrics`
     - Evaluation metrics, including AUC, GAUC and other commonly used metrics in recommendation systems
   * - :doc:`hooks`
     - Hooks, including log printing, performance analysis tools and other hooks
   * - :doc:`optim`
     - Optimizer module, provides sparse parameter optimization
   * - :doc:`framework`
     - Training framework, provides trainer and checkpoint management
   * - :doc:`serialize`
     - Serialization module, includes Saver, Loader and Checkpoint direct reading tools
   * - :doc:`utils`
     - Utility functions, including logging, data processing and other auxiliary functions

Quick Index
-----------

**Common Classes and Functions**

- :class:`recis.io.OrcDataset` - ORC Dataset
- :class:`recis.features.FeatureEngine` - Feature Engine
- :class:`recis.nn.EmbeddingEngine` - Embedding Engine
- :class:`recis.metrics.auroc.AUROC` - AUC Calculation
- :class:`recis.optim.sparse_adamw.SparseAdamW` - Sparse AdamW Optimizer
- :class:`recis.framework.trainer.Trainer` - Trainer

**Important Configuration Classes**

- :class:`recis.nn.EmbeddingOption` - Embedding Configuration
- :class:`recis.framework.trainer.TrainingArguments` - Training Parameter Configuration

**Checkpoint Reading Tools**

- :class:`recis.serialize.checkpoint_reader.CheckpointReader` - Read Checkpoint Files

Version Compatibility
---------------------

.. note::
   
   RecIS requires Python 3.10+ and PyTorch 2.4+. APIs may differ between versions, please refer to the documentation for the corresponding version.
