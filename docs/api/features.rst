Feature Processing Module
=========================

RecIS's Feature Processing module provides efficient and flexible feature engineering and preprocessing capabilities, supporting complex feature transformation pipelines and operator fusion optimization for high-performance feature processing solutions.

Core Features
-------------

**High-Performance Feature Processing**
   - **Operator Fusion Optimization**: Automatically identifies fusible operators and performs batch processing for significant performance improvements
   - **GPU Accelerated Computing**: Support for CUDA-accelerated core operators including hashing, bucketing, and cutoff

**Feature Operators**
   - **Hash Operators**: Provide FarmHash and MurmurHash algorithms for large-scale categorical feature processing
   - **Bucketing Operators**: Support for numerical feature discretization and boundary bucketing
   - **Sequence Processing**: Provide sequence truncation, padding, and length control functionality
   - **Feature Crossing**: Support for multi-feature cross combinations to generate new features

**Flexible Execution Engine**
   - **Dynamic Compilation**: Dynamic compilation and optimization of feature pipelines
   - **Caching Mechanism**: Feature computation result caching and reuse

Operator Fusion Optimization
----------------------------

RecIS's feature processing module provides advanced operator fusion optimization mechanisms:

**Automatic Fusion Recognition**
   - Automatically identifies fusible operators of the same type
   - Batch processing improves GPU utilization
   - Reduces memory copying and kernel launch overhead

**Supported Fusion Operators**
   - **FusedHashOP**: Batch hash processing
   - **FusedBoundaryOP**: Batch bucketing processing
   - **FusedModOP**: Batch modulo operation processing
   - **FusedCutoffOP**: Batch sequence truncation processing

.. currentmodule:: recis.features

Core Components
---------------

FeatureEngine
~~~~~~~~~~~~~

.. autoclass:: FeatureEngine
   :members: __init__, forward

Feature
~~~~~~~

.. currentmodule:: recis.features.feature

.. autoclass:: Feature
   :members: __init__, add_op, forward

Feature Operations
------------------

.. currentmodule:: recis.features.op

Basic Operations
~~~~~~~~~~~~~~~~

SelectField
^^^^^^^^^

.. autoclass:: SelectField
   :members: __init__, forward

SelectFields
^^^^^^^^^^^^^^

.. autoclass:: SelectFields
   :members: __init__, forward

Hash Operations
~~~~~~~~~~~~~~~

Hash
^^^^^^^^^^^

.. autoclass:: Hash
   :members: __init__, forward

IDMultiHash
^^^^^^^^^^^

.. autoclass:: IDMultiHash
   :members: __init__, forward

Integer Modulo
~~~~~~~~~~~~~~

.. autoclass:: Mod
   :members: __init__, forward

Float Bucketing Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Bucketize
   :members: __init__

Sequence Truncation
~~~~~~~~~~~~~~~~~~~

.. autoclass:: SequenceTruncate
   :members: __init__, forward

Cross Features
~~~~~~~~~~~~~~

FeatureCross
^^^^^^^^^^^^^^^

.. autoclass:: FeatureCross
   :members: __init__, forward

Advanced Usage
--------------

**Custom Operations**

You can inherit from base operation classes to implement custom feature processing:

.. code-block:: python

   from recis.features.op import _OP
   
   class CustomNormalize(_OP):
       def __init__(self, mean=0.0, std=1.0):
           super().__init__()
           self.mean = mean
           self.std = std
       
       def forward(self, x):
           return (x - self.mean) / self.std
   
   # Use custom operation
   custom_feature = Feature("normalized_score").\
                        add_op(SelectField("score")),\
                        add_op(CustomNormalize(mean=0.5, std=0.2))
