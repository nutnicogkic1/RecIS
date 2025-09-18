基础使用
========

本章节介绍 RecIS 的基础使用方法，帮助初学者快速上手。

环境准备
--------

在开始之前，确保已经正确安装了 recis 以及数据拓展 column-io：

.. code-block:: python

   import torch
   import recis
   import column_io

基础概念
--------

**FeatureEngine**
  特征处理引擎，支持复杂的特征变换

**HashTable**
  稀疏参数存储的核心组件

**DynamicEmbedding**
  动态扩张的 embedding 表，支持稀疏参数的自动管理

**EmbeddingOption**
  Embedding表配置选项

**EmbeddingEngine**
  Embedding表管理引擎，管理多张DynamicEmbedding，并提供稀疏合并等优化策略

**SparseOptimizer**
  专为稀疏参数设计的优化器

**Trainer**
  训练管理器

数据处理
-----------

数据转换工具
~~~~~~~~~~~~

**将CSV文件转换成ORC格式数据**

.. code-block:: python

   from recis.nn import DynamicEmbedding, EmbeddingOption
   
读取数据示例
~~~~~~~~~~~~

.. code-block:: python

    import os
    from recis.io.orc_dataset import OrcDataset

    worker_idx = int(os.environ.get("RANK", 0))
    worker_num = int(os.environ.get("WORLD_SIZE", 1))
    dataset = OrcDataset(
        1024, # batch size
        worker_idx=worker_idx,
        worker_num=worker_num,
        read_threads_num=2, # 读取数据线程数
        prefetch=1, # 预取数据个数
        is_compressed=False,
        drop_remainder=True, # 删除不满batch的数据
        transform_fn=[lambda x: x[0]],
        dtype=torch.float32,
        device="cuda", # dataset数据结果直接输出到cuda上
        save_interval=None,
    )
    data_paths = ["./data_dir/"]
    for path in data_paths:
        dataset.add_path(path)
    # 读取定长特征
    dataset.fixedlen_feature("label", [0.0])
    # 读取变长特征
    dataset.varlen_feature("user_id")
    dataset.varlen_feature("item_id")

    # 构建读取数据
    iter = iter(dataset)
    data = next(iter)
    

特征工程
-----------

**基础特征处理**

.. code-block:: python

   from recis.features import FeatureEngine
   from recis.features.feature import Feature
   from recis.features.op import SelectField, Hash, Bucketize
   
   # 定义特征处理流水线
   features = [
       # 用户 ID 哈希
       Feature(
           name="user_id",
           ops=[
               SelectField("user_id"),
               Hash(bucket_size=100000)
           ]
       ),
       
       # 商品 ID 哈希
       Feature(
           name="item_id",
           ops=[
               SelectField("item_id"),
               Hash(bucket_size=50000)
           ]
       ),
       
       # 年龄分桶
       Feature(
           name="age_bucket",
           ops=[
               SelectField("age"),
               Bucketize(boundaries=[18, 25, 35, 45, 55, 65])
           ]
       )
   ]
   
   # 创建特征引擎
   feature_engine = FeatureEngine(features)
   
   # 处理数据
   input_data = {
       'user_id': torch.LongTensor([[1], [2], [3]]),
       'item_id': torch.LongTensor([[101], [102], [103]]),
       'age': torch.FloatTensor([[25], [35], [45]])
   }
   
   processed_data = feature_engine(input_data)
   
   print("原始数据:", input_data)
   print("处理后数据:", processed_data)

稀疏Embedding表
---------------

构建Embedding表
~~~~~~~~~~~~~~~

**创建第一个 Embedding**

.. code-block:: python

   from recis.nn import DynamicEmbedding, EmbeddingOption
   
   # 配置 embedding 选项
   emb_opt = EmbeddingOption(
       embedding_dim=64,
       shared_name="my_embedding",
       combiner="sum"
   )
   
   # 创建动态 embedding
   embedding = DynamicEmbedding(emb_opt)
   
   # 使用 embedding
   ids = torch.LongTensor([[1], [2], [3], [100], [1000]])
   emb_output = embedding(ids)
   
   print(f"输入 ID: {ids}")
   print(f"Embedding 输出形状: {emb_output.shape}")
   print(f"Embedding 输出: {emb_output}")

**使用EmbeddingEngine管理优化Embedding表**

.. code-block:: python

    from recis.nn import EmbeddingEngine, EmbeddingOption
    
    # 配置 embedding 选项
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
    
    # 创建动态 embedding
    embedding = EmbeddingEngine(
        {"user_emb": user_emb_opt, "item_emb": id_emb_opt}
    )
    
    # 使用 embedding
    user_ids = torch.LongTensor([[1], [2], [3], [100], [1000]])
    item_ids = torch.LongTensor([[11], [22], [33], [111], [1111]])
    emb_output = embedding({"user_emb": user_ids, "item_emb": item_ids})
    
    print(f"Embedding 输出: {emb_output}")

构建稀疏参数优化器
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from recis.optim import SparseAdamW
   from recis.nn.modules.hashtable import filter_out_sparse_param
   
   # 创建一个简单模型
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
   
   # 分离稀疏和稠密参数
   sparse_params = filter_out_sparse_param(model)
   
   print("稀疏参数:", list(sparse_params.keys()))
   
   # 创建优化器
   sparse_optimizer = SparseAdamW(sparse_params, lr=0.001)
   dense_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

使用Trainer进行训练
-------------------

简单训练
~~~~~~~~~

.. code-block:: python

    # 构建模型
    # model = ...
    # 构建数据
    # dataset = ...
    # 定义优化器
    # sparse_params = filter_out_sparse_param(model)

    # sparse_optim = SparseAdamWTF(sparse_params, lr=0.001)
    # opt = AdamW(params=model.parameters(), lr=0.001)

    train_config = TrainingArguments(
        gradient_accumulation_steps=1,
        output_dir="./ckpt/",
        log_steps=10,
        train_steps=100, # 只训练100step
    )

    trainer = Trainer(
        model=model,
        args=train_config,
        train_dataset=dataset,
        dense_optimizers=(opt, None),
        sparse_optimizer=sparse_optim,
    )
    trainer.train()

高阶训练拓展
~~~~~~~~~~~~

**1. 边训练边测试**

.. code-block:: python

    # 构建模型
    # model = ...
    # 构建数据
    # dataset = ...
    # 定义优化器
    # sparse_params = filter_out_sparse_param(deepctrmodel_model.state_dict())

    # sparse_optim = SparseAdamWTF(sparse_params, lr=0.001)
    # opt = AdamW(params=model.parameters(), lr=0.001)

    train_config = TrainingArguments(
        gradient_accumulation_steps=1,
        output_dir="./ckpt/",
        log_steps=10,
        train_steps=100,
        eval_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=train_config,
        train_dataset=dataset,
        dense_optimizers=(opt, None),
        sparse_optimizer=sparse_optim,
    )
    # 训练100step，测试10step，这样的循环模式重复10次
    trainer.train_and_evaluate(10) 

**2. 自定义训练流程**

.. code-block:: python

    from framework.trainer import Trainer
    class MyTrainer(Trainer):
         def _train_step(self, data, epoch, metrics):
            self.dense_optimizer.zero_grad()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.zero_grad()
            loss = self.model(data)
            metrics.update(epoch=epoch)
            metrics.update(loss=loss)
            metrics.update(get_global_metrics())
            loss.backward()
            self.dense_optimizer.step()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()
            if self.dense_lr_scheduler is not None:
                self.dense_lr_scheduler.step()

**3. 自定义Saver保存信息**

.. code-block:: python

    from framework.trainer import Trainer
    from framework.checkpoint_manager import Saver

    class MySaver(Saver):
        def save_dense_params(self, ckpt_path: str, dense_state_dict: OrderedDict):
            pt_file = os.path.join(ckpt_path, "model.pt")
            with fs.open(pt_file, "wb") as f:
                torch.save(dense_state_dict, f=f)

    class MyTrainer(Trainer):
         def build_saver(self, model, args):
            saver = MySaver(
                model,
                self.sparse_optimizer,
                output_dir=args.output_dir,
                max_keep=args.max_to_keep,
                concurrency=args.save_concurrency_per_rank,
            )
            return saver

评估指标使用
------------

**基础指标计算**

.. code-block:: python

   from recis.metrics import AUROC
   
   # 创建 AUC 指标
   auc_metric = AUROC(num_thresholds=200)
   
   # 模拟预测和标签
   predictions = torch.rand(1000)  # 随机预测值 [0, 1]
   labels = torch.randint(0, 2, (1000,))  # 随机标签 {0, 1}
   
   # 更新指标
   auc_metric.update(predictions, labels)
   
   # 计算 AUC
   auc_score = auc_metric.compute()
   print(f"AUC Score: {auc_score:.4f}")
   
   # 重置指标
   auc_metric.reset()

**在训练中使用指标**

.. code-block:: python

   from recis.framework.metrics import add_metric
   
   class ModelWithMetrics(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.embedding = DynamicEmbedding(EmbeddingOption(embedding_dim=32))
           self.linear = torch.nn.Linear(32, 1)
           self.auc_metric = AUROC(num_thresholds=200)
           self.loss_fn = torch.nn.BCEWithLogitsLoss()
       
       def forward(self, batch):
           ids = batch['ids']
           labels = batch['labels']
           
           # 模型预测
           emb = self.embedding(ids)
           logits = self.linear(emb).squeeze()
           
           # 计算损失
           loss = self.loss_fn(logits, labels)
           
           # 更新指标
           probs = torch.sigmoid(logits)
           self.auc_metric.update(probs, labels.long())
           auc = self.auc_metric.compute()
           
           # 添加指标到训练框架
           add_metric("auc", auc)
           add_metric("loss", loss)
           
           return loss
   
   # 使用带指标的模型
   model_with_metrics = ModelWithMetrics()

下一步
------

完成基础使用后，您可以：

1. 学习 :doc:`deepfm_example` 了解DeepFM完整模型实现
2. 学习 :doc:`seq2seq_example` 了解SEq2Seq完整模型实现
3. 学习 :doc:`ctr_example` 了解更多特征转换场景
4. 参考 :doc:`../api/index` 深入了解 API 详情

如果遇到问题，请查看 :doc:`../faq` 或寻求社区帮助。