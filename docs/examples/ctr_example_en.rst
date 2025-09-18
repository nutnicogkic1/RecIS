CTR Example Model
================

This chapter introduces the basic usage methods of RecIS through a simple CTR model, helping beginners get started quickly and learn more feature processing usage.
For specific training code, please refer to the examples/deepctr directory

Data Construction
-----------------

**Import Required Modules**

.. code-block:: python

    import os
    import random
    import string

    import numpy as np
    import pyarrow as pa
    import pyarrow.orc as orc

**Generate Data**

.. code-block:: python

    # Data output directory, data information definition
    file_dir = "./fake_data/"
    os.makedirs(file_dir, exist_ok=True)
    bs = 2047
    file_num = 10

    dense1 = np.random.rand(bs, 8)
    dense2 = np.random.rand(bs, 1)
    label = np.floor(np.random.rand(bs, 1) + 0.5, dtype=np.float32)
    sparse1 = np.arange(bs, dtype=np.int64).reshape(bs, 1)
    sparse2 = np.arange(bs, dtype=np.int64).reshape(bs, 1)
    sparse3 = np.arange(bs, dtype=np.int64).reshape(bs, 1)

    # Generate long sequence features
    long_int_seq = []
    for i in range(bs):
        seq_len = np.random.randint(1, 2000, dtype=np.int64)
        sequence = np.random.randint(0, 1000000, size=seq_len, dtype=np.int64).tolist()
        long_int_seq.append(sequence)

    def generate_random_string(length):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choices(characters, k=length))

    # Generate long string sequence features
    strs = []
    for i in range(1000):
        strs.append(generate_random_string(10))
    long_str_seq = []
    for i in range(bs):
        seq_len = np.random.randint(1, 2000, dtype=np.int64)
        sequence = random.choices(strs, k=seq_len)
        long_str_seq.append(sequence)

    data = {
        "label": label.tolist(),
        "dense1": dense1.tolist(),
        "dense2": dense2.tolist(),
        "sparse1": sparse1.tolist(),
        "sparse2": sparse2.tolist(),
        "sparse3": sparse3.tolist(),
        "sparse4": long_int_seq,
        "sparse5": long_str_seq,
    }

    table = pa.Table.from_pydict(data)
    # Generate data
    for i in range(file_num):
        orc.write_table(table, os.path.join(file_dir, "data_{}.orc".format(i)))

Data Definition
---------------

**Define IO Parameters**

.. code-block:: python

    from dataclasses import dataclass
    @dataclass
    class IOArgs:
        data_paths: str
        batch_size: int
        # Concurrency for data reading
        thread_num: int
        # Data prefetch quantity
        prefetch: int
        drop_remainder: bool

**Build Dataset**

.. code-block:: python

    import os
    import torch

    import recis
    from recis.io.orc_dataset import OrcDataset

    def get_dataset(io_args):
        # Get current rank id and rank num in distributed mode for data parallelism
        worker_idx = int(os.environ.get("RANK", 0))
        worker_num = int(os.environ.get("WORLD_SIZE", 1))
        dataset = OrcDataset(
            io_args.batch_size,
            worker_idx=worker_idx,
            worker_num=worker_num,
            read_threads_num=io_args.thread_num,
            prefetch=io_args.prefetch,
            is_compressed=False,
            drop_remainder=io_args.drop_remainder,
            # Data preprocessing
            transform_fn=[lambda x: x[0]],
            dtype=torch.float32,
            # Batch packaging results directly placed on cuda
            device="cuda",
            save_interval=None,
        )
        data_paths = io_args.data_paths.split(",")
        for path in data_paths:
            dataset.add_path(path)
        # Set feature columns to read
        # Read fixed-length features and default values
        dataset.fixedlen_feature("label", [0.0])
        dataset.fixedlen_feature("dense1", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dataset.fixedlen_feature("dense2", [0.0])
        # Read variable-length features
        dataset.varlen_feature("sparse1")
        dataset.varlen_feature("sparse2")
        dataset.varlen_feature("sparse3")
        dataset.varlen_feature("sparse4")
        # sparse5 is a string sequence that needs hash processing.
        # You can perform hash operations in the dataset by setting hash_type="farm",
        # or by setting hash_type=None and trans_int8=True to read strings
        # as int8 byte streams, then perform hashing through HashOp later.
        dataset.varlen_feature("sparse5", hash_type=None, trans_int8=True)
        return dataset

Feature Processing Configuration
--------------------------------

.. code-block:: python

    from recis.features.feature import Feature
    from recis.features.op import (
        Bucketize,
        SelectField,
        SelectFields,
        FeatureCross,
        SequenceTruncate,
        Mod,
    )
    def get_feature_conf():
        feature_confs = []
        # dense1 feature read directly, dim is 8
        feature_confs.append(Feature("dense1").add_op(SelectField("dense1", dim=8)))
        # dense2 feature, dim is 1, needs bucketing transformation
        feature_confs.append(
            Feature("dense2")
            .add_op(SelectField("dense2", dim=1))
            .add_op(Bucketize([0, 0.5, 1]))
        )
        # sparse1 / sparse2 features, read directly
        feature_confs.append(Feature("sparse1").add_op(SelectField("sparse1")))
        feature_confs.append(Feature("sparse2").add_op(SelectField("sparse2")))
        # sparse3 feature, perform modulo 10000 calculation
        feature_confs.append(
            Feature("sparse3").add_op(SelectField("sparse3")).add_op(Mod(10000))
        )
        # sparse4 feature, perform modulo calculation and truncation
        feature_confs.append(
            Feature("sparse4")
                .add_op(SelectField("sparse4"))
                .add_op(Mod(10000))
                .add_op(SequenceTruncate(seq_len=1000,
                                        truncate=True,
                                        truncate_side="right",
                                        check_length=True,
                                        n_dims=2))
        )
        # sparse5 feature, perform hash, modulo and truncation
        feature_confs.append(
            Feature("sparse5")
                .add_op(SelectField("sparse5"))
                .add_op(Hash(hash_type="farm"))
                .add_op(Mod(10000))
                .add_op(SequenceTruncate(seq_len=1000,
                                        truncate=True,
                                        truncate_side="right",
                                        check_length=True,
                                        n_dims=2))
        )
        # sparse1_x_sparse2 feature, perform feature crossing
        feature_confs.append(
            Feature("sparse1_x_sparse2")
            .add_op(SelectFields([SelectField("sparse1"), SelectField("sparse2")]))
            .add_op(FeatureCross())
            .add_op(Mod(1000))
        )
        return feature_confs

Embedding Configuration
-----------------------

.. code-block:: python

    from recis.nn.initializers import Initializer, TruncNormalInitializer
    from recis.nn.modules.embedding import EmbeddingOption
    def get_embedding_conf():
        emb_conf = {}
        # dense2 feature looks up emb table with dim=8, name=sparse1
        emb_conf["dense2"] = EmbeddingOption(
            embedding_dim=8,
            shared_name="sparse1",
            combiner="sum",
            initializer=TruncNormalInitializer(mean=0, std=0.01),
            device=torch.device("cuda"),
        )
        # sparse1 feature looks up emb table with dim=8, name=sparse1 (shares the same emb table with dense2)
        emb_conf["sparse1"] = EmbeddingOption(
            embedding_dim=8,
            shared_name="sparse1",
            combiner="sum",
            initializer=TruncNormalInitializer(mean=0, std=0.01),
            device=torch.device("cuda"),
        )
        # sparse2 feature looks up emb table with dim=16, name=sparse2
        emb_conf["sparse2"] = EmbeddingOption(
            embedding_dim=16,
            shared_name="sparse2",
            combiner="sum",
            initializer=TruncNormalInitializer(mean=0, std=0.01),
            device=torch.device("cuda"),
        )
        # sparse3 feature looks up emb table with dim=8, name=sparse3
        emb_conf["sparse3"] = EmbeddingOption(
            embedding_dim=8,
            shared_name="sparse3",
            combiner="sum",
            initializer=TruncNormalInitializer(mean=0, std=0.01),
            device=torch.device("cuda"),
        )
        emb_conf["sparse4"] = EmbeddingOption(
            embedding_dim=16,
            shared_name="sparse4",
            combiner="tile",
            initializer=TruncNormalInitializer(mean=0, std=0.01),
            device=torch.device("cuda"),
            combiner_kwargs={"tile_len": 1000}
        )
        emb_conf["sparse5"] = EmbeddingOption(
            embedding_dim=16,
            shared_name="sparse5",
            combiner="tile",
            initializer=TruncNormalInitializer(mean=0, std=0.01),
            device=torch.device("cuda"),
            combiner_kwargs={"tile_len": 1000}
        )
        emb_conf["sparse1_x_sparse2"] = EmbeddingOption(
            embedding_dim=16,
            shared_name="sparse1_x_sparse2",
            combiner="mean",
            initializer=TruncNormalInitializer(mean=0, std=0.01),
            device=torch.device("cuda"),
        )
        return emb_conf

Model definition
-----------------

**Define sparse model**

.. code-block:: python

    import torch
    import torch.nn as nn

    from recis.features.feature_engine import FeatureEngine
    from recis.nn import EmbeddingEngine

    class SparseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_engine = FeatureEngine(feature_list=get_feature_conf())
            self.embedding_engine = EmbeddingEngine(get_embedding_conf())

        def forward(self, samples: dict):
            samples = self.feature_engine(samples)
            samples = self.embedding_engine(samples)
            labels = samples.pop("label")
            return samples, labels

**Define dense model**

.. code-block:: python

    class DenseModel(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            layers.extend([nn.Linear(8 + 8 + 8 + 16 + 8 + 16000 + 16000 + 16, 128), nn.ReLU()])
            layers.extend([nn.Linear(128, 64), nn.ReLU()])
            layers.extend([nn.Linear(64, 32), nn.ReLU()])
            layers.extend([nn.Linear(32, 1)])
            self.dnn = nn.Sequential(*layers)

        def forward(self, x):
            x = self.dnn(x)
            logits = torch.sigmoid(x)
            return logits

**Define whole model**

.. code-block:: python

    from recis.framework.metrics import add_metric
    from recis.metrics.auroc import AUROC

    class DeepCTR(nn.Module):
        def __init__(self):
            super().__init__()
            self.sparse_arch = SparseModel()
            self.dense_arch = DenseModel()
            self.auc_metric = AUROC(num_thresholds=200, dist_sync_on_step=True)
            self.loss = nn.BCELoss()

        def forward(self, samples: dict):
            samples, labels = self.sparse_arch(samples)
            dense_input = torch.cat(
                [
                    samples["dense1"],
                    samples["dense2"],
                    samples["sparse1"],
                    samples["sparse2"],
                    samples["sparse3"],
                    samples["sparse4"],
                    samples["sparse5"],
                    samples["sparse1_x_sparse2"],
                ],
                -1,
            )
            logits = self.dense_arch(dense_input)
            
            loss = self.loss(logits.squeeze(), labels.squeeze())
            
            self.auc_metric.update(logits.squeeze(), labels.squeeze())
            auc_score = self.auc_metric.compute()
            
            add_metric("auc", auc_score)
            add_metric("loss", loss)
            
            return loss

Training
----------

**Define training process**

.. code-block:: python

    import os
    import torch
    from torch.optim import AdamW

    from recis.framework.trainer import Trainer, TrainingArguments
    from recis.nn.modules.hashtable import HashTable, filter_out_sparse_param
    from recis.optim import SparseAdamWTF
    from recis.utils.logger import Logger

    logger = Logger(__name__)

    def train():
        deepctr_model = DeepCTR()
        # get dataset
        train_dataset = get_dataset(
            io_args=IOArgs(
                data_paths="./fake_data/",
                batch_size=1024,
                thread_num=1,
                prefetch=1,
                drop_remainder=True,
            ),
        )
        logger.info(str(deepctr_model))
        sparse_params = filter_out_sparse_param(deepctr_model)

        sparse_optim = SparseAdamWTF(sparse_params, lr=0.001)
        opt = AdamW(params=deepctr_model.parameters(), lr=0.001)

        train_config = TrainingArguments(
            gradient_accumulation_steps=1,
            output_dir="./ckpt/",
            model_bank=None,
            log_steps=10,
            train_steps=100,
            train_epoch=1,
            eval_steps=None,
            save_steps=1000,
            max_to_keep=3,
            save_concurrency_per_rank=2,
        )

        deepctr_model = deepctr_model.cuda()
        trainer = Trainer(
            model=deepctr_model,
            args=train_config,
            train_dataset=train_dataset,
            dense_optimizers=(opt, None),
            sparse_optimizer=sparse_optim,
            data_to_cuda=False,
        )
        trainer.train()


**Set parallelism parameters (optional)**

.. code-block:: python

    import os
    from multiprocessing import cpu_count

    def set_num_threads():
        cpu_num = cpu_count() // 16
        os.environ["OMP_NUM_THREADS"] = str(cpu_num)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
        os.environ["MKL_NUM_THREADS"] = str(cpu_num)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
        torch.set_num_interop_threads(cpu_num)
        torch.set_num_threads(cpu_num)
        # set device for local run
        torch.cuda.set_device(int(os.getenv("RANK", "-1")))

**Set random seed (optional)**

.. code-block:: python

    import numpy as np
    import random

    def set_seed(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        np.random.seed(seed)
        random.seed(seed)

**Main entry**

.. code-block:: python

    import torch.distributed as dist
    if __name__ == "__main__":
        set_num_threads()
        set_seed(42)
        dist.init_process_group()
        train()

Start training
---------------

.. code-block:: shell

    export PYTHONPATH=$PWD
    MASTER_PORT=12455
    WORLD_SIZE=2
    ENTRY=deepctr.py

    torchrun --nproc_per_node=$WORLD_SIZE --master_port=$MASTER_PORT $ENTRY
