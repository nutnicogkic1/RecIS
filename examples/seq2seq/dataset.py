import os
from dataclasses import dataclass

import torch
from feature_config import FEATURE_CONFIG, SEQ_LEN

from recis.features.feature import Feature
from recis.features.op import SelectField, SequenceTruncate
from recis.io.orc_dataset import OrcDataset
from recis.nn.initializers import TruncNormalInitializer
from recis.nn.modules.embedding import EmbeddingOption


@dataclass
class IOArgs:
    data_paths: str
    batch_size: int
    thread_num: int
    prefetch: int
    drop_remainder: bool


def get_dataset(io_args):
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
    for item in FEATURE_CONFIG:
        fn = item["name"]
        dataset.varlen_feature(
            fn, item.get("hash_type", None), item.get("hash_bucket_size", 0)
        )
    return dataset


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
