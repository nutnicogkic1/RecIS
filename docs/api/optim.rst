Optimizer Module
================

RecIS's optimizer module is specifically designed for sparse parameter optimization, providing efficient sparse parameter update algorithms.

.. currentmodule:: recis.optim

Sparse Optimizers
-----------------

SparseAdamW
~~~~~~~~~~~

.. currentmodule:: recis.optim.sparse_adamw

.. autoclass:: SparseAdamW
   :members: __init__, step, zero_grad, set_grad_accum_steps

SparseAdam
~~~~~~~~~~
.. currentmodule:: recis.optim.sparse_adam

.. autoclass:: SparseAdam
   :members: __init__, step, zero_grad, set_grad_accum_steps

TensorFlow Compatible Optimizers
--------------------------------

SparseAdamWTF
~~~~~~~~~~~~~

.. currentmodule:: recis.optim.sparse_adamw_tf

.. autoclass:: SparseAdamWTF
   :members: __init__, step, zero_grad, set_grad_accum_steps

AdamWTF
~~~~~~~
.. currentmodule:: recis.optim.adamw_tf

.. autoclass:: AdamWTF
   :members: __init__, step, zero_grad

Usage Guide
-----------

**Basic Usage Flow**

1. **Parameter Separation**:

   .. code-block:: python

      from recis.nn.modules.hashtable import filter_out_sparse_param
      
      # Separate sparse and dense parameters
      sparse_params = filter_out_sparse_param(model)

2. **Create Optimizers**:

   .. code-block:: python

      from recis.optim import SparseAdamW
      from torch.optim import AdamW
      
      # Sparse parameter optimizer
      sparse_optimizer = SparseAdamW(sparse_params, lr=0.001)
      
      # Dense parameter optimizer
      dense_optimizer = AdamW(model.parameters(), lr=0.001)

Performance Optimization Recommendations
----------------------------------------

**Parameter Tuning**

1. **Learning Rate Settings**:

   .. code-block:: python

      # Sparse parameters usually need larger learning rates
      sparse_optimizer = SparseAdamW(sparse_params, lr=0.01)
      dense_optimizer = AdamW(model.parameters(), lr=0.001)

2. **Weight Decay**:

   .. code-block:: python

      # Adjust weight decay based on model size
      sparse_optimizer = SparseAdamW(
          sparse_params, 
          lr=0.001,
          weight_decay=0.01  # Larger models can use larger weight decay

3. **Beta Parameter Adjustment**:

   .. code-block:: python

      # For sparse updates, beta parameters can be adjusted
      sparse_optimizer = SparseAdamW(
          sparse_params,
          lr=0.001,
          beta1=0.9,   # First moment estimate
          beta2=0.999  # Second moment estimate
      )

Frequently Asked Questions
--------------------------

**Q: What's the difference between sparse optimizers and regular optimizers?**

A: Sparse optimizers are specifically designed for HashTable parameters with the following characteristics:
- Only update parameters with gradients
- Support dynamic parameter expansion
- More efficient memory usage
- Compatible with distributed training

**Q: How to choose the right optimizer?**

A: Selection recommendations:
- For sparse embeddings: Use SparseAdamW
- For dense layers: Use standard AdamW
- For TensorFlow alignment: Use TF version optimizers