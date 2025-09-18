import os
import random
from dataclasses import dataclass
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.distributed as dist
from dataset import IOArgs, get_dataset
from deepfm_model import DeepFM
from torch.optim import AdamW

from recis.framework.trainer import Trainer, TrainingArguments
from recis.hooks import ProfilerHook
from recis.nn.modules.hashtable import filter_out_sparse_param
from recis.optim.sparse_adamw import SparseAdamW
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
    parser.add_arguments(IOArgs, dest="dataset")
    return parser.parse_args()


def train(args):
    model = DeepFM()
    logger.info(str(model))

    # get dataset
    dataset = get_dataset(args.dataset)

    # optimizer
    sparse_params = filter_out_sparse_param(model)
    logger.info(f"Hashtables: {sparse_params}")
    # hashtable use sparse optimizer
    sparse_optim = SparseAdamW(sparse_params, lr=args.lr.sparse_lr)
    # dense module use normal optimizer
    opt = AdamW(params=model.parameters(), lr=args.lr.dense_lr)

    model = model.cuda()

    # hooks and trainer
    trainer = Trainer(
        model=model,
        args=args.train_config,
        train_dataset=dataset,
        dense_optimizers=(opt, None),
        sparse_optimizer=sparse_optim,
    )

    if int(os.environ.get("RANK", 0)) == 0:
        hooks = [
            ProfilerHook(
                wait=1,
                warmup=20,
                active=1,
                repeat=4,
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
