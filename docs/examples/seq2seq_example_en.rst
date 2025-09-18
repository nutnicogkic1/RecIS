Seq2Seq Example Model
=====================

This chapter demonstrates how to train a seq2seq model using RecIS, helping beginners get familiar with the basic usage of RecIS. The related code is located in the examples/seq2seq directory.

Data Construction
-----------------

The ORC data for training is already prepared in the examples/seq2seq/data directory. You can preview this data using Python's pyarrow library:

.. code-block:: python

    import pyarrow.orc as orc
    file = "./data/part-0.orc"
    table = orc.ORCFile(file).read()
    print(table)

By printing the table, you can see that this is sequence feature data containing user behavior sequences, item IDs, category IDs, and other information.

Data Definition
---------------

**Define IO Parameters**

.. code-block:: python

    @dataclass
    class IOArgs:
        data_paths: str
        batch_size: int
        # Number of concurrent threads for data reading
        thread_num: int
        # Number of data prefetch
        prefetch: int
        drop_remainder: bool

**Build Dataset**

.. code-block:: python

    def get_dataset(io_args):
        # Parallel data acquisition
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
            transform_fn=[lambda x: x[0]],
            dtype=torch.float32,
            device="cuda",
            save_interval=1000,
        )
        data_paths = io_args.data_paths.split(",")
        for path in data_paths:
            dataset.add_path(path)
        
        # Read variable-length features
        for item in FEATURE_CONFIG:
            fn = item["name"]
            dataset.varlen_feature(
                fn, item.get("hash_type", None), item.get("hash_bucket_size", 0)
            )
        return dataset

Feature Processing Configuration
--------------------------------

.. code-block:: python

    # Feature processing configuration
    def get_feature_conf():
        feature_confs = []
        for item in FEATURE_CONFIG:
            fn = item["name"]
            feature_confs.append(
                Feature(fn)
                .add_op(SelectField(fn))
                .add_op(
                    SequenceTruncate(
                        seq_len=SEQ_LEN,
                        truncate=True,
                        truncate_side="right",
                        check_length=False,
                        n_dims=3,
                        dtype=torch.int64,
                    )
                )
            )
        return feature_confs

The `FEATURE_CONFIG` defines the basic information of features:

.. code-block:: python

    FEATURE_CONFIG = [
        {
            "name": "item_id",
            "emb_dim": 128,
            "hash_bucket_size": 2048000,
            "shard_name": "item_id",
        },
        {
            "name": "cate_id",
            "emb_dim": 128,
            "hash_bucket_size": 2048,
            "shard_name": "cate_id",
        },
        {
            "name": "behavior_id",
            "emb_dim": 128,
            "hash_type": "murmur",
            "hash_bucket_size": 0,
            "shard_name": "behavior_id",
        },
        {
            "name": "timestamp",
            "emb_dim": 128,
            "hash_type": "murmur",
            "hash_bucket_size": 2048000,
            "shard_name": "timestamp",
        },
    ]

Embedding Configuration
-----------------------

.. code-block:: python

    def get_embedding_conf():
        emb_conf = {}
        for item in FEATURE_CONFIG:
            fn = item["name"]
            emb_dim = item.get("emb_dim", 0)
            shard_name = item.get("shard_name", fn)
            emb_conf[fn] = EmbeddingOption(
                embedding_dim=emb_dim,
                shared_name=shard_name,
                combiner="mean",
                initializer=TruncNormalInitializer(std=0.001),
                device=torch.device("cuda"),
            )
        return emb_conf

Model Definition
----------------

**Define Sparse Model**

.. code-block:: python

    class SparseModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Feature processing
            self.feature_engine = FeatureEngine(feature_list=get_feature_conf())
            # Calculate feature embeddings
            self.embedding_engine = EmbeddingEngine(get_embedding_conf())

        def forward(self, samples: dict):
            samples = self.feature_engine(samples)
            samples = self.embedding_engine(samples)
            return samples

**Define Transformer Encoder**

.. code-block:: python

    class CasualMultiHeadAttention(nn.Module):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            self.attn = nn.Linear(config.hidden_size, 3 * config.hidden_size)
            self.proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.dropout)
            self.nhead = config.nhead
            self.d_head = config.hidden_size // self.nhead
            self.hidden_size = config.hidden_size

        def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
            batch, seq, _ = x.size()
            q, k, v = self.attn(x).split(self.hidden_size, dim=2)
            q = q.view(batch, seq, self.nhead, self.d_head).transpose(1, 2)
            k = k.view(batch, seq, self.nhead, self.d_head).transpose(1, 2)
            v = v.view(batch, seq, self.nhead, self.d_head).transpose(1, 2)

            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=True
            )

            attn_output = attn_output.transpose(1, 2).contiguous().view(*x.size())
            attn_output = self.dropout(attn_output)
            output = self.proj(attn_output)
            output = self.dropout(output)
            return output

    class FeedForward(nn.Module):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(config.hidden_size, config.dim_feedforward),
                nn.GELU(),
                nn.Linear(config.dim_feedforward, config.hidden_size),
                nn.Dropout(config.dropout),
            )

        def forward(self, x: torch.Tensor):
            return self.net(x)

    class TransformerEncoderLayer(nn.Module):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.norm1 = nn.LayerNorm(config.hidden_size)
            self.attn = CasualMultiHeadAttention(config)
            self.norm2 = nn.LayerNorm(config.hidden_size)
            self.ffn = FeedForward(config)

        def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
            x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
            x = x + self.ffn(self.norm2(x))
            return x

    class Transformer(nn.Module):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.layers = nn.ModuleList(
                [TransformerEncoderLayer(config) for _ in range(config.num_layers)]
            )
            self.final_layer_norm = nn.LayerNorm(config.hidden_size)
            self.seq_len = config.seq_len

        def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
            for layer in self.layers:
                x = layer(x, attn_mask=attn_mask)
            x = self.final_layer_norm(x)
            return x

**Define Decoder**

.. code-block:: python

    class Decoder(nn.Module):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.proj = nn.Linear(config.emb_size, config.hidden_size)
            self.trans = Transformer(config)
            self.loss_fn = nn.CrossEntropyLoss()

        def cal_loss(self, preds: torch.Tensor, items: torch.Tensor):
            preds = preds[:, 1:, :]
            preds = preds.reshape(-1, preds.shape[-1])
            items = items[:, :-1, :]
            items = items.reshape(-1, items.shape[-1])
            preds = F.normalize(preds, p=2, dim=-1, eps=1e-6)
            items = F.normalize(items, p=2, dim=-1, eps=1e-6)
            labels = torch.arange(preds.shape[0], device=preds.device, dtype=torch.long)
            cos_sim = torch.matmul(preds, items.t())
            loss = self.loss_fn(cos_sim, labels)
            with torch.no_grad():
                hits = (cos_sim.detach().argmax(dim=1) == labels).sum()
            add_metric("hit_rate", hits / preds.shape[0])
            add_metric("loss", loss)
            return loss

        def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
            items = self.proj(x)
            preds = self.trans(items, attn_mask)
            if self.training:
                loss = self.cal_loss(preds, items)
                return loss
            else:
                return preds

**Define Complete Model**

.. code-block:: python

    class Seq2SeqModel(nn.Module):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.dense = Decoder(config)
            self.sparse = SparseModel()
            self.casual_mask = (
                torch.tril(torch.ones(config.seq_len, config.seq_len))
                .view(1, 1, config.seq_len, config.seq_len)
                .cuda()
            )

        def build_embedding(self, samples: dict[torch.Tensor]):
            embs = []
            for item in FEATURE_CONFIG:
                fn = item["name"]
                embs.append(samples[fn])
            return torch.cat(embs, dim=-1)

        def cal_mask(self, seq_len):
            return self.casual_mask[:, :, :seq_len, :seq_len]

        def forward(self, samples: dict[torch.Tensor]):
            samples = self.sparse(samples)
            emb = self.build_embedding(samples)
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                return self.dense(emb, self.cal_mask(emb.shape[1]))

The `ModelConfig` defines the basic configuration of the model:

.. code-block:: python

    @dataclass
    class ModelConfig:
        seq_len: int = 1024
        hidden_size: int = 1024
        num_layers: int = 8
        nhead: int = 8
        dim_feedforward: int = 1024
        dropout: float = 0.1
        emb_size: int = 512

Training Entry
--------------

**Define Training Process**

First, get the dataset:

.. code-block:: python

    train_dataset = get_dataset(args.io_args)

Then create the model and separate optimizers for sparse and dense models:

.. code-block:: python

    model = Seq2SeqModel(args.model_config)
    model = model.cuda()

    # optimizer
    sparse_param = filter_out_sparse_param(model)
    dense_opt, sparse_opt = get_optimizer(model, args.lr.dense_lr, args.lr.sparse_lr)

The `get_optimizer` function is defined as follows:

.. code-block:: python

    def get_optimizer(model: nn.Module, dense_lr, sparse_lr):
        sparse_param = filter_out_sparse_param(model)
        dense_opt = AdamW(model.parameters(), lr=dense_lr)
        sparse_opt = SparseAdamW(sparse_param, lr=sparse_lr)
        return (dense_opt, sparse_opt)

Finally, create the training process:

.. code-block:: python

    trainer = Trainer(
        model=model,
        args=args.train_config,
        train_dataset=train_dataset,
        dense_optimizers=(dense_opt, None),
        sparse_optimizer=sparse_opt,
    )

**Environment Setup**

Set up distributed environment and random seeds:

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
        # set device for local run
        torch.cuda.set_device(int(os.getenv("RANK", "-1")))

    def set_seed(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        np.random.seed(seed)
        random.seed(seed)

**Start Training**

Start training through the `run.sh` script: `bash run.sh`.
