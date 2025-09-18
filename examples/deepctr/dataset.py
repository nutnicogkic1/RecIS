import os
from dataclasses import dataclass

import torch

from recis.features.feature import Feature
from recis.features.op import (
    Bucketize,
    FeatureCross,
    Hash,
    Mod,
    SelectField,
    SelectFields,
    SequenceTruncate,
)
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
    dataset.fixedlen_feature("label", [0.0])
    dataset.fixedlen_feature("dense1", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dataset.fixedlen_feature("dense2", [0.0])
    dataset.varlen_feature("sparse1")
    dataset.varlen_feature("sparse2")
    dataset.varlen_feature("sparse3")
    dataset.varlen_feature("sparse4")
    # sparse5 is a string sequence that needs to be hashed.
    # You can perform hashing directly in the dataset using hash_type="farm".
    # Alternatively, you can first convert the strings to a byte stream using hash_type=None
    # and trans_int8=True, and then apply the hash using HashOp.
    dataset.varlen_feature("sparse5", hash_type=None, trans_int8=True)
    return dataset


def get_feature_conf():
    feature_confs = []
    feature_confs.append(Feature("dense1").add_op(SelectField("dense1", dim=8)))
    feature_confs.append(
        Feature("dense2")
        .add_op(SelectField("dense2", dim=1))
        .add_op(Bucketize([0, 0.5, 1]))
    )
    feature_confs.append(Feature("sparse1").add_op(SelectField("sparse1")))
    feature_confs.append(Feature("sparse2").add_op(SelectField("sparse2")))
    feature_confs.append(
        Feature("sparse3").add_op(SelectField("sparse3")).add_op(Mod(10000))
    )
    feature_confs.append(
        Feature("sparse4")
        .add_op(SelectField("sparse4"))
        .add_op(Mod(10000))
        .add_op(
            SequenceTruncate(
                seq_len=1000,
                truncate=True,
                truncate_side="right",
                check_length=True,
                n_dims=2,
            )
        )
    )
    feature_confs.append(
        Feature("sparse5")
        .add_op(SelectField("sparse5"))
        .add_op(Hash(hash_type="farm"))
        .add_op(Mod(10000))
        .add_op(
            SequenceTruncate(
                seq_len=1000,
                truncate=True,
                truncate_side="right",
                check_length=True,
                n_dims=2,
            )
        )
    )
    feature_confs.append(
        Feature("sparse1_x_sparse2")
        .add_op(SelectFields([SelectField("sparse1"), SelectField("sparse2")]))
        .add_op(FeatureCross())
        .add_op(Mod(1000))
    )
    return feature_confs


def get_embedding_conf():
    emb_conf = {}
    emb_conf["dense2"] = EmbeddingOption(
        embedding_dim=8,
        shared_name="sparse1",
        combiner="sum",
        initializer=TruncNormalInitializer(mean=0, std=0.01),
        device=torch.device("cuda"),
    )
    emb_conf["sparse1"] = EmbeddingOption(
        embedding_dim=8,
        shared_name="sparse1",
        combiner="sum",
        initializer=TruncNormalInitializer(mean=0, std=0.01),
        device=torch.device("cuda"),
    )
    emb_conf["sparse2"] = EmbeddingOption(
        embedding_dim=16,
        shared_name="sparse2",
        combiner="sum",
        initializer=TruncNormalInitializer(mean=0, std=0.01),
        device=torch.device("cuda"),
    )
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
        combiner_kwargs={"tile_len": 1000},
    )
    emb_conf["sparse5"] = EmbeddingOption(
        embedding_dim=16,
        shared_name="sparse5",
        combiner="tile",
        initializer=TruncNormalInitializer(mean=0, std=0.01),
        device=torch.device("cuda"),
        combiner_kwargs={"tile_len": 1000},
    )
    emb_conf["sparse1_x_sparse2"] = EmbeddingOption(
        embedding_dim=16,
        shared_name="sparse1_x_sparse2",
        combiner="mean",
        initializer=TruncNormalInitializer(mean=0, std=0.01),
        device=torch.device("cuda"),
    )
    return emb_conf
