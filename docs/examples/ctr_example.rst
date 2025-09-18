CTR示例模型
===========

本章节将通过一个简单的CTR模型介绍 RecIS 的基础使用方法，帮助初学者快速上手，并了解更多特征处理用法。
具体训练代码可参考 examples/deepctr 目录

数据构建
------------

**导入需要模块**

.. code-block:: python

    import os
    import random
    import string

    import numpy as np
    import pyarrow as pa
    import pyarrow.orc as orc

**生成数据**

.. code-block:: python

    # 数据产出目录，数据信息定义
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

    # 生成长序列特征
    long_int_seq = []
    for i in range(bs):
        seq_len = np.random.randint(1, 2000, dtype=np.int64)
        sequence = np.random.randint(0, 1000000, size=seq_len, dtype=np.int64).tolist()
        long_int_seq.append(sequence)

    def generate_random_string(length):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choices(characters, k=length))

    # 生成长字符串序列特征
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
    # 生成数据
    for i in range(file_num):
        orc.write_table(table, os.path.join(file_dir, "data_{}.orc".format(i)))

数据定义
----------

**定义IO参数**

.. code-block:: python

    from dataclasses import dataclass
    @dataclass
    class IOArgs:
        data_paths: str
        batch_size: int
        # 读取数据的并发数
        thread_num: int
        # 数据预取数量
        prefetch: int
        drop_remainder: bool

**构建dataset**

.. code-block:: python

    import os
    import torch

    import recis
    from recis.io.orc_dataset import OrcDataset

    def get_dataset(io_args):
        # 获取当前分布式模式下的rank id和rank num，用于数据并行
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
            # 数据预处理
            transform_fn=[lambda x: x[0]],
            dtype=torch.float32,
            # batch打包结果直接place到cuda上
            device="cuda",
            save_interval=None,
        )
        data_paths = io_args.data_paths.split(",")
        for path in data_paths:
            dataset.add_path(path)
        # 设定需要读取的特征列
        # 读取定长特征，以及默认值
        dataset.fixedlen_feature("label", [0.0])
        dataset.fixedlen_feature("dense1", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dataset.fixedlen_feature("dense2", [0.0])
        # 读取变长特征
        dataset.varlen_feature("sparse1")
        dataset.varlen_feature("sparse2")
        dataset.varlen_feature("sparse3")
        dataset.varlen_feature("sparse4")
        # sparse5 是一个需要进行哈希处理的字符串序列。
        # 可以在数据集中通过设置 hash_type="farm" 来执行哈希操作，
        # 或者通过设置 hash_type=None 和 trans_int8=True 来将字符串
        # 读取为 int8 的字节流，之后再通过 HashOp 执行哈希。
        dataset.varlen_feature("sparse5", hash_type=None, trans_int8=True)
        return dataset

特征处理配置
------------

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
        # dense1特征直接读取，dim为8
        feature_confs.append(Feature("dense1").add_op(SelectField("dense1", dim=8)))
        # dense2特征，dim为1，需要做分桶转换
        feature_confs.append(
            Feature("dense2")
            .add_op(SelectField("dense2", dim=1))
            .add_op(Bucketize([0, 0.5, 1]))
        )
        # sparse1 / sparse2特征，直接读取
        feature_confs.append(Feature("sparse1").add_op(SelectField("sparse1")))
        feature_confs.append(Feature("sparse2").add_op(SelectField("sparse2")))
        # sparse3特征，做10000的取模计算处理
        feature_confs.append(
            Feature("sparse3").add_op(SelectField("sparse3")).add_op(Mod(10000))
        )
        # sparse4特征，做取模计算和截断处理
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
        # sparse5特征，做哈希、取模和截断处理
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
        # sparse1_x_sparse2特征，做特征交叉
        feature_confs.append(
            Feature("sparse1_x_sparse2")
            .add_op(SelectFields([SelectField("sparse1"), SelectField("sparse2")]))
            .add_op(FeatureCross())
            .add_op(Mod(1000))
        )
        return feature_confs

Embedding配置
-----------------

.. code-block:: python

    from recis.nn.initializers import Initializer, TruncNormalInitializer
    from recis.nn.modules.embedding import EmbeddingOption
    def get_embedding_conf():
        emb_conf = {}
        # dense2特征查找dim=8，name=sparse1的emb表
        emb_conf["dense2"] = EmbeddingOption(
            embedding_dim=8,
            shared_name="sparse1",
            combiner="sum",
            initializer=TruncNormalInitializer(mean=0, std=0.01),
            device=torch.device("cuda"),
        )
        # sparse1特征查找dim=8，name=sparse1的emb表(和dense2共用同一张emb表)
        emb_conf["sparse1"] = EmbeddingOption(
            embedding_dim=8,
            shared_name="sparse1",
            combiner="sum",
            initializer=TruncNormalInitializer(mean=0, std=0.01),
            device=torch.device("cuda"),
        )
        # sparse2特征查找dim=16，name=sparse2的emb表
        emb_conf["sparse2"] = EmbeddingOption(
            embedding_dim=16,
            shared_name="sparse2",
            combiner="sum",
            initializer=TruncNormalInitializer(mean=0, std=0.01),
            device=torch.device("cuda"),
        )
        # sparse3特征查找dim=8，name=sparse3的emb表
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

模型定义
----------

**定义稀疏部分模型**

.. code-block:: python

    import torch
    import torch.nn as nn

    from recis.features.feature_engine import FeatureEngine
    from recis.nn import EmbeddingEngine

    class SparseModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 构建特征处理引擎
            self.feature_engine = FeatureEngine(feature_list=get_feature_conf())
            # 构建Embedding处理引擎
            self.embedding_engine = EmbeddingEngine(get_embedding_conf())

        def forward(self, samples: dict):
            samples = self.feature_engine(samples)
            samples = self.embedding_engine(samples)
            labels = samples.pop("label")
            return samples, labels

**定义稠密部分模型**

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

**定义完整模型**

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
            
            # 计算损失
            loss = self.loss(logits.squeeze(), labels.squeeze())
            
            # 更新AUC指标
            self.auc_metric.update(logits.squeeze(), labels.squeeze())
            auc_score = self.auc_metric.compute()
            
            # 添加指标到训练框架
            add_metric("auc", auc_score)
            add_metric("loss", loss)
            
            return loss

训练入口
----------

**定义训练流程**

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


**设定并发参数（可选）**

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

**设定随机种子（可选）**

.. code-block:: python

    import numpy as np
    import random

    def set_seed(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        np.random.seed(seed)
        random.seed(seed)

**主脚本**

.. code-block:: python

    import torch.distributed as dist
    if __name__ == "__main__":
        set_num_threads()
        set_seed(42)
        # 创建通信组
        dist.init_process_group()
        train()

开始训练
----------

.. code-block:: shell

    export PYTHONPATH=$PWD
    MASTER_PORT=12455
    # 分布式规模
    WORLD_SIZE=2
    # 入口脚本
    ENTRY=deepctr.py

    torchrun --nproc_per_node=$WORLD_SIZE --master_port=$MASTER_PORT $ENTRY
