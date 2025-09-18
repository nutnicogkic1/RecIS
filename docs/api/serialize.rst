Parameter Saving and Loading
============================

Save Format
-----------

Model Files
~~~~~~~~~~~

N files with the suffix .safetensors will be generated, where N is the number of GPU multiplied by the parallel thread count set when saving the model. The file format is the same as safetensor.

Auxiliary Files
~~~~~~~~~~~~~~~

Some intermediate files will be generated when saving the model to facilitate debugging when saving or loading models encounters bugs.

1. index file: This file format is the same as json file format. It has two keys, the first key is block_index, which records which model file the tensor is saved to; the second key is file_index, which records the index of the model file.
2. tensorkey.json file: Records the conversion relationship of tensor_name when saving the model, such as user_id/part_0_2, where 2 indicates double GPU training, and 0 indicates that the current tensor is assigned to GPU 0
3. torch_rank_weights_embs_table_multi_shard.json, this json file will specifically describe all tensors, as shown below:

.. code-block:: bash

  "feature_layer.feature_ebc.ebc.embedding_bags.gbdt_cvr.embedding/part_0_1": {
        "name": "feature_layer.feature_ebc.ebc.embedding_bags.gbdt_ctr.embedding/part_0_1",              # tensor name
        "dense": false,                                                                                  # non-dense tensor
        "dimension": 16,                                                                                 # dimension
        "dtype": "float32",                                                                              # data type
        "hashmap_key": "feature_layer.feature_ebc.ebc.embedding_bags.gbdt_ctr.id/part_0_1",              # key and value in hash table
        "hashmap_value": "feature_layer.feature_ebc.ebc.embedding_bags.gbdt_ctr.embedding/part_0_1",
        "shape": [
            1211,
            16
        ],
        "is_hashmap": true
    },

    # This is dense tensor, same meaning as above
    "feature_layer.feature_ebc.ebc.embedding_bags.buyer_star_name.weight": {
            "name": "feature_layer.feature_ebc.ebc.embedding_bags.buyer_star_name.weight",
            "dense": false,
            "dimension": 4,
            "dtype": "float32",
            "shape": [
                64,
                4
            ],
            "is_hashmap": false
        },

Save and Load Interface
-----------------------

CheckpointReader
~~~~~~~~~~~~~~~~

.. currentmodule:: recis.serialize.checkpoint_reader

.. autoclass:: CheckpointReader
   :members: __init__, tensor_names, read_tensor, tensor_shape, tensor_dtype

Saver
~~~~~

.. currentmodule:: recis.serialize.saver

.. autoclass:: Saver
   :members: __init__, save

Loader
~~~~~~

.. currentmodule:: recis.serialize.loader

.. autoclass:: Loader
    :members: __init__, load