import os
import random
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.distributed as dist
from dataset import IOArgs, get_dataset
from model import DeepCTR
from torch.optim import AdamW

from recis.framework.trainer import Trainer, TrainingArguments
from recis.nn.modules.hashtable import filter_out_sparse_param
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
    train()
