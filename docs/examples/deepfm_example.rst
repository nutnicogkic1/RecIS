DeepFM示例模型
==============

本章节使用 RecIS 训练 deepfm 模型，帮助初学者熟悉 RecIS 的基础使用方法。相关代码在 examples/deepfm 目录下。

数据构建
------------

训练使用的 orc 数据已经准备好，在 examples/deepfm/data 目录下。可以使用 python 的 pyarrow 库预览这些数据：

.. code-block:: python

    import pyarrow.orc as orc
    file = "./data/part-0.orc"
    table = orc.ORCFile(file).read()
    print(table)

打印 table，可以发现这是一些 id 以及特征类型：

.. code-block:: bash

    c1: list<item: int64>
        child 0, item: int64
    ...
    i1: list<item: double>
        child 0, item: double

对应到 examples/deepfm/feature_config.py 文件，numeric 下的特征为 double 类型，categorical 下的特征为 int64 类型。

数据定义
----------

**定义IO参数**

.. code-block:: python

    @dataclass
    class IOArgs:
        data_paths: str
        batch_size: int
        # 读取数据的并发数
        thread_num: int
        # 数据预取数量
        prefetch: int
        drop_remainder: bool

**构建 dataset**

.. code-block:: python

    def transform_batch(batch_list):
        # 数据转换处理
        result_batch = {}
        result_batch = {"label": batch_list[0]["label"]}
        for fn in FEATURES["numeric"]:
            result_batch[fn] = batch_list[0][fn]

        # 一个特征产出 emb 和 bias 两个内容
        for fn in FEATURES["categorical"]:
            result_batch[fn + "_emb"] = batch_list[0][fn]
            result_batch[fn + "_bias"] = batch_list[0][fn]
        return result_batch

    def get_dataset(args):
        # 并行获取数据
        worker_idx = int(os.environ.get("RANK", 0))
        worker_num = int(os.environ.get("WORLD_SIZE", 1))
        dataset = OrcDataset(
            args.batch_size,
            worker_idx=worker_idx,
            worker_num=worker_num,
            read_threads_num=args.thread_num,
            prefetch=args.prefetch,
            is_compressed=False,
            drop_remainder=args.drop_remainder,
            transform_fn=transform_batch,
            dtype=torch.float32,
            device="cuda",
            save_interval=1000,
        )
        data_paths = args.data_paths.split(",")
        dataset.add_paths(data_paths)

        # 读取定长特征，以及默认值
        dataset.fixedlen_feature("label", [0.0])

        # 读取变长特征
        for fn in FEATURES["numeric"] + FEATURES["categorical"]:
            dataset.varlen_feature(fn)
        return dataset

特征处理配置
------------

.. code-block:: python

    # 特征处理
    # add op 指处理特征，这里的特征经过 SelectField op，仅拿出 key 值
    def get_feature_conf():
        feature_confs = []
        # numeric 特征直接读取，dim 为 1
        for fn in FEATURES["numeric"]:
            feature_confs.append(
                Feature(fn)
                .add_op(SelectField(fn, dim=1))
            )
        # 添加 categorical 特征
        for fn in FEATURES["categorical"]:
            for si, suffix in enumerate(["_emb", "_bias"]):
                real_fn = fn + suffix
                feature_confs.append(
                    Feature(real_fn)
                    .add_op(SelectField(real_fn))
                )
        return feature_confs

Embedding 配置
-----------------

.. code-block:: python

    def get_embedding_conf():
        emb_conf = {}
        for fn in FEATURES["categorical"]:
            # 为每个特征创建各自的 embedding 表
            for si, suffix in enumerate(["_emb", "_bias"]):
                real_fn = fn + suffix
                emb_conf[real_fn] = EmbeddingOption(
                    embedding_dim=EMBEDDING_DIM if si == 0 else 1,
                    shared_name=real_fn,
                    combiner="sum",
                    initializer=TruncNormalInitializer(std=0.001),
                    device=torch.device("cuda"),
                )
        return emb_conf

模型定义
----------

**定义稀疏部分模型**

.. code-block:: python

    class SparseModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 特征处理
            self.feature_engine = FeatureEngine(feature_list=get_feature_conf())
            # 计算特征的 embedding
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
            self.embedding_dim = EMBEDDING_DIM
            num_numeric_features = len(FEATURES["numeric"])
            num_categorical_features = len(FEATURES["categorical"])

            # Initialize weights and biases for numerical features
            self.numeric_embs = nn.Parameter(
                torch.randn(num_numeric_features, self.embedding_dim) * 0.001
            )
            self.numeric_biases = nn.Parameter(torch.randn(num_numeric_features, 1) * 0.001)

            # Build the DNN part
            layers = []
            dnn_structure = DNN_SHAPE
            input_size = (
                num_numeric_features + num_categorical_features
            ) * self.embedding_dim
            for size in dnn_structure:
                layers.extend([nn.Linear(input_size, size), nn.ReLU()])
                input_size = size
            layers.append(nn.Linear(input_size, 1))  # Output layer
            self.dnn = nn.Sequential(*layers)


        def forward(self, samples, labels):
            batch_size = labels.shape[0]

            # Get numeric features
            numeric_embeddings = self.numeric_embs.repeat(batch_size, 1, 1)
            numeric_biases = self.numeric_biases.repeat(batch_size, 1, 1)
            numeric_weights = torch.stack(
                [samples[fn] for fn in FEATURES["numeric"]],
                dim=1,
            )

            # Get embeddings and biases for categorical features
            category_embeddings = torch.stack(
                [samples[f"{fn}_emb"] for fn in FEATURES["categorical"]],
                dim=1,
            )
            category_biases = torch.stack(
                [samples[f"{fn}_bias"] for fn in FEATURES["categorical"]],
                dim=1,
            )
            category_weights = torch.ones(
                [batch_size, len(FEATURES["categorical"]), 1], device=category_biases.device
            )

            # Merge all feature embeddings and biases
            all_embeddings = torch.cat([numeric_embeddings, category_embeddings], dim=1)
            all_biases = torch.cat([numeric_biases, category_biases], dim=1)
            all_weights = torch.cat([numeric_weights, category_weights], dim=1)

            # Calculate first-order effects
            first_order_output = torch.sum(
                torch.squeeze(all_weights * all_biases, dim=-1), dim=-1, keepdim=True
            )

            # Calculate second-order effects
            squared_sum = torch.sum(all_embeddings * all_weights, dim=1) ** 2
            sum_squared = torch.sum((all_embeddings**2) * (all_weights**2), dim=1)
            second_order_output = 0.5 * torch.sum(
                squared_sum - sum_squared, dim=-1, keepdim=True
            )

            # DNN output
            dnn_input = all_embeddings.view(
                -1, all_embeddings.shape[1] * all_embeddings.shape[2]
            )
            dnn_output = self.dnn(dnn_input)

            # Final output
            final_output = torch.sigmoid(
                first_order_output + second_order_output + dnn_output
            )
            return final_output

**定义完整模型**

.. code-block:: python

    class DeepFM(nn.Module):
        def __init__(self):
            super(DeepFM, self).__init__()
            self.sparse_model = SparseModel()
            self.dense_model = DenseModel()
            self.loss_function = nn.BCELoss()
            self.auc_metric = AUROC(num_thresholds=200, dist_sync_on_step=True)

        def forward(self, samples):
            samples, labels = self.sparse_model(samples)
            final_output = self.dense_model(samples, labels)
            # Calculate loss
            loss = self.loss_function(final_output, labels)

            self.auc_metric.update(final_output, labels)
            auc = self.auc_metric.compute()
            add_metric("auc", auc)

            return loss

训练入口
----------

**定义训练流程**

首先获取数据集：

.. code-block:: python

    dataset = get_dataset(args.dataset)

然后创建模型，在分别创建稀疏模型和稠密模型的优化器

.. code-block:: python

    model = DeepFM()
    model = model.cuda()

    # optimizer
    sparse_params = filter_out_sparse_param(model)
    logger.info(f"Hashtables: {sparse_params}")
    # hashtable use sparse optimizer
    sparse_optim = SparseAdamW(sparse_params, lr=args.lr.sparse_lr)
    # dense module use normal optimizer
    opt = AdamW(params=model.parameters(), lr=args.lr.dense_lr)

最后创建训练流程

.. code-block:: python

    # hooks and trainer
    trainer = Trainer(
        model=model,
        args=args.train_config,
        train_dataset=dataset,
        dense_optimizers=(opt, None),
        sparse_optimizer=sparse_optim,
    )

其中 `args.train_config` 在 `examples/deepfm/run.sh` 中指定：

.. code-block:: bash

    ARG="--data_paths=./data/              \
        --batch_size=1000                  \
        --thread_num=1                     \
        --prefetch=1                       \
        --drop_remainder=true              \
        --gradient_accumulation_steps=4    \
        --output_dir="./ckpt"              \
        --log_steps=10                     \
        --save_steps=2000 "

**环境设置**

设置分布式相关环境和随机种子：

.. code-block:: python

    def set_num_threads():
        cpu_num = cpu_count() // 16
        os.environ["OMP_NUM_THREADS"] = str(cpu_num)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
        os.environ["MKL_NUM_THREADS"] = str(cpu_num)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
        torch.set_num_interop_threads(cpu_num)
        torch.set_num_threads(cpu_num)
        torch.cuda.set_device(int(os.getenv("RANK", "-1")))

    def set_seed(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            # For multi-GPU setups
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

**开始训练**

通过 `run.sh` 脚本启动训练：`bash run.sh` 即可。