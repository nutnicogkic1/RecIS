import os
from dataclasses import dataclass

import torch
from feature_config import EMBEDDING_DIM, FEATURES

from recis.features.feature import Feature
from recis.features.op import SelectField
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


def transform_batch(batch_list):
    # transform raw batch from odps
    result_batch = {}
    result_batch = {"label": batch_list[0]["label"]}
    for fn in FEATURES["numeric"]:
        result_batch[fn] = batch_list[0][fn]

    for fn in FEATURES["categorical"]:
        result_batch[fn + "_emb"] = batch_list[0][fn]
        result_batch[fn + "_bias"] = batch_list[0][fn]
    return result_batch


def get_dataset(args):
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

    # add feature
    dataset.fixedlen_feature("label", [0.0])

    for fn in FEATURES["numeric"] + FEATURES["categorical"]:
        dataset.varlen_feature(fn)
    return dataset


def get_feature_conf():
    feature_confs = []
    for fn in FEATURES["numeric"]:
        feature_confs.append(Feature(fn).add_op(SelectField(fn, dim=1)))
    for fn in FEATURES["categorical"]:
        for si, suffix in enumerate(["_emb", "_bias"]):
            real_fn = fn + suffix
            feature_confs.append(Feature(real_fn).add_op(SelectField(real_fn)))
    return feature_confs


def get_embedding_conf():
    emb_conf = {}
    for fn in FEATURES["categorical"]:
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
