HashTable Module
================

RecIS's HashTable is the core structure of dynamic embedding tables. It supports dynamic expansion of embedding tables and provides feature admission and feature filtering capabilities.

HashTable
----------

.. currentmodule:: recis.nn.modules.hashtable

.. autoclass:: HashTable
   :members: __init__, forward, accept_grad, grad, clear_grad, insert, ids, embeddings, slot_group, children_info

Utility Functions
-----------------

.. currentmodule:: recis.nn.modules.hashtable

.. autofunction:: filter_out_sparse_param

.. autofunction:: gen_slice