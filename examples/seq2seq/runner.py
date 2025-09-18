import os
import random
from dataclasses import dataclass
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from dataset import IOArgs, get_dataset
from seq2seq_model import ModelConfig, Seq2SeqModel
from torch.optim import AdamW

from recis.framework.trainer import Trainer, TrainingArguments
from recis.hooks import ProfilerHook
from recis.nn.modules.hashtable import filter_out_sparse_param
from recis.optim import SparseAdamW
from recis.utils.logger import Logger
from recis.utils.parser import ArgumentParser


logger = Logger(__name__)


@dataclass
class LRConfig:
    dense_lr: float = 0.001
    sparse_lr: float = 0.001


def parse_args():
    parser = ArgumentParser()
    parser.add_arguments(LRConfig, dest="lr")
    parser.add_arguments(TrainingArguments, dest="train_config")
    parser.add_arguments(IOArgs, dest="io_args")
    parser.add_arguments(ModelConfig, dest="model_config")
    return parser.parse_args()


def get_optimizer(model: nn.Module, dense_lr, sparse_lr):
    sparse_param = filter_out_sparse_param(model)
    dense_opt = AdamW(model.parameters(), lr=dense_lr)
    sparse_opt = SparseAdamW(sparse_param, lr=sparse_lr)
    return (dense_opt, sparse_opt)


def train(args):
    model = Seq2SeqModel(args.model_config)
    logger.info(str(model))

    train_dataset = get_dataset(args.io_args)

    # optimizer
    dense_opt, sparse_opt = get_optimizer(model, args.lr.dense_lr, args.lr.sparse_lr)

    model = model.cuda()

    trainer = Trainer(
        model=model,
        args=args.train_config,
        train_dataset=train_dataset,
        dense_optimizers=(dense_opt, None),
        sparse_optimizer=sparse_opt,
    )

    # timeline for debug
    if int(os.environ.get("RANK", 0)) == 0:
        hooks = [
            ProfilerHook(
                wait=1,
                warmup=100,
                active=1,
                repeat=2,
                output_dir=args.train_config.output_dir,
            )
        ]
        trainer.add_hooks(hooks)

    trainer.train()


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


if __name__ == "__main__":
    set_num_threads()
    set_seed(42)
    dist.init_process_group()
    args = parse_args()
    logger.info(f"Input args : {args}")
    train(args)
