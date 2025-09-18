from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
)
from torch import nn
from torch.utils.data import Dataset

from recis.framework.checkpoint_manager import CheckpointManager, Saver
from recis.framework.metrics import get_global_metrics
from recis.hooks import Hook, LoggerHook
from recis.optim import sparse_optim
from recis.utils.data_utils import copy_data_to_device
from recis.utils.logger import Logger


logger = Logger(__name__)


@dataclass
class TrainingArguments:
    """Configuration class for training parameters.

    This dataclass contains all the configuration parameters needed for training,
    including optimization settings, logging intervals, and checkpoint management.

    Attributes:
        gradient_accumulation_steps (int): Number of steps to accumulate gradients
                                         before performing an optimizer step. Defaults to 1.
        output_dir (str): Directory where checkpoints and logs will be saved.
                         Defaults to "output_dir".
        model_bank (Optional[list]): List of model bank paths for initialization.
                                   Defaults to None.
        log_steps (int): Number of training steps between logging. Defaults to 100.
        train_steps (Optional[int]): Maximum number of training steps. If None,
                                   will train for full epochs. Defaults to None.
        train_epoch (Optional[int]): Number of training epochs. Defaults to 1.
        eval_steps (Optional[int]): Number of evaluation steps. If None, evaluates
                                  on full dataset. Defaults to None.
        save_steps (Optional[int]): Number of steps between checkpoint saves.
                                  Defaults to 1000.
        max_to_keep (int): Maximum number of checkpoints to keep. Defaults to 5.
        save_concurrency_per_rank (int): Number of concurrent save operations per rank.
                                        Defaults to 4.
    """

    gradient_accumulation_steps: int = 1
    output_dir: str = "output_dir"
    model_bank: Optional[list] = None
    log_steps: int = 100
    train_steps: Optional[int] = None
    train_epoch: Optional[int] = 1
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = 1000
    max_to_keep: int = 5
    save_concurrency_per_rank: int = 4


class Trainer:
    """Main training orchestrator with distributed training and checkpoint management.

    The Trainer class provides a comprehensive training framework that handles:
    - Distributed training coordination using Accelerate
    - Automatic checkpoint saving and loading
    - Training and evaluation loops with metrics tracking
    - Hook system for extensible training workflows
    - Support for both dense and sparse optimizers

    Attributes:
        args (TrainingArguments): Training configuration parameters.
        hooks (List[Hook]): List of training hooks for extensibility.
        train_dataset (Optional[Dataset]): Training dataset.
        eval_dataset (Optional[Dataset]): Evaluation dataset.
        model (nn.Module): The model to train.
        dense_optimizer (torch.optim.Optimizer): Dense parameter optimizer.
        dense_lr_scheduler: Learning rate scheduler for dense optimizer.
        sparse_optimizer (Optional[sparse_optim.SparseOptimizer]): Sparse parameter optimizer.
        data_to_cuda (bool): Whether to automatically move data to CUDA.
        accelerator (Accelerator): Accelerate instance for distributed training.
        checkpoint_manager (CheckpointManager): Handles checkpoint operations.

    Example:

    .. code-block:: python

        from recis.framework import Trainer, TrainingArguments
        from recis.optim import SparseAdamW
        from torch.optim import AdamW

        # Set training arguments
        training_args = TrainingArguments(
            output_dir="./checkpoints",
            train_steps=10000,
            eval_steps=1000,
            save_steps=2000,
            log_steps=100,
            gradient_accumulation_steps=4,
        )

        # split sparse params
        from recis.nn.modules.hashtable import filter_out_sparse_param

        sparse_params = filter_out_sparse_param(model)

        # create optimizers
        sparse_optimizer = SparseAdamW(sparse_params, lr=0.001)
        dense_optimizer = AdamW(model.parameters(), lr=0.001)

        # create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dense_optimizers=(dense_optimizer, None),
            sparse_optimizer=sparse_optimizer,
            data_to_cuda=True,
        )

        # train the model
        trainer.train()
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        hooks: Optional[List[Hook]] = None,
        dense_optimizers: Tuple[
            torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR
        ] = (None, None),
        sparse_optimizer: Optional[sparse_optim.SparseOptimizer] = None,
        data_to_cuda: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the Trainer with model, datasets, and training configuration.

        Args:
            model (Optional[nn.Module]): The model to train.
            args (TrainingArguments): Training configuration. If None, uses default.
            train_dataset (Optional[Dataset]): Training dataset.
            eval_dataset (Optional[Dataset]): Evaluation dataset.
            hooks (Optional[List[Hook]]): List of training hooks for extensibility.
            dense_optimizers (Tuple): Tuple of (optimizer, lr_scheduler) for dense parameters.
            sparse_optimizer (Optional[sparse_optim.SparseOptimizer]): Optimizer for sparse parameters.
            data_to_cuda (bool): Whether to automatically move data to CUDA. Defaults to False.
            **kwargs: Additional arguments passed to Accelerator.
        """
        if hooks is None:
            hooks = []
        if args is None:
            args = TrainingArguments()
        self.args = args
        self.hooks = hooks
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.model = model
        self.dense_optimizer = dense_optimizers[0]
        self.dense_lr_scheduler = dense_optimizers[1]
        self.sparse_optimizer = sparse_optimizer
        self.data_to_cuda = data_to_cuda
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs, init_kwargs],
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            **kwargs,
        )
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        (
            self.model,
            self.dense_optimizer,
            self.dense_lr_scheduler,
        ) = self.accelerator.prepare(
            self.model, self.dense_optimizer, self.dense_lr_scheduler
        )
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.set_grad_accum_steps(self.gradient_accumulation_steps)
        self._global_step = torch.scalar_tensor(0, dtype=torch.int64)
        self.build_checkpoint_manager(model, args)
        self.has_restore = False

        self.hooks.append(LoggerHook(self.args.log_steps))
        self.stop_state = torch.scalar_tensor(0, dtype=torch.int64).cuda()

    def build_checkpoint_manager(self, model, args):
        saver = self.build_saver(model, args)
        self.checkpoint_manager = CheckpointManager(
            saver=saver, save_interval=args.save_steps
        )
        if self.train_dataset is not None:
            saver.register_io_state("train_io", self.train_dataset)
            if hasattr(self.train_dataset, "_window_paths"):
                saver.register_for_checkpointing("train_window_io", self.train_dataset)
        if self.eval_dataset is not None and hasattr(
            self.eval_dataset, "_window_paths"
        ):
            saver.register_io_state("eval_io", self.eval_dataset)
            if hasattr(self.eval_dataset, "_window_paths"):
                saver.register_for_checkpointing("eval_window_io", self.eval_dataset)
        if self.dense_optimizer is not None:
            saver.register_for_checkpointing("dense_optimizer", self.dense_optimizer)

    def build_saver(self, model, args):
        saver = Saver(
            model,
            self.sparse_optimizer,
            output_dir=args.output_dir,
            max_keep=args.max_to_keep,
            concurrency=args.save_concurrency_per_rank,
        )
        return saver

    def add_hooks(self, hooks: List[Hook]):
        """Add multiple hooks to the trainer.

        Args:
            hooks (List[Hook]): List of hooks to add.
        """
        for hook in hooks:
            self.add_hook(hook)

    def add_hook(self, hook: Hook):
        """Add a single hook to the trainer.

        Args:
            hook (Hook): The hook to add.
        """
        self.hooks.append(hook)

    def restore(self):
        """Restore model and training state from checkpoints.

        This method loads the latest checkpoint if available and restores
        the model state, optimizer state, and training progress.
        """
        if not self.has_restore:
            self.checkpoint_manager.load_model_bank(self.args.model_bank)
            self._global_step = self.checkpoint_manager.restore()
            self.has_restore = True

    def train(self, train_steps=None):
        """Execute the training loop.

        Args:
            train_steps (Optional[int]): Override for number of training steps.
                                       If None, uses args.train_steps.
        """
        self.restore()
        for epoch in range(self.args.train_epoch):
            if hasattr(self.train_dataset, "_window_paths"):
                self.train_dataset.reset()
            self._train_loop(
                self.args.train_steps if train_steps is None else train_steps,
                epoch=epoch,
            )
        for hook in self.hooks:
            hook.end()

    def train_and_evaluate(self, epochs=1, train_steps=None, eval_steps=None):
        """Execute alternating training and evaluation loops.

        Args:
            epochs (int): Number of epochs to train. Defaults to 1.
            train_steps (Optional[int]): Override for number of training steps per epoch.
            eval_steps (Optional[int]): Override for number of evaluation steps.
        """
        self.restore()
        for epoch in range(epochs):
            self._train_loop(
                self.args.train_steps if train_steps is None else train_steps,
                epoch=epoch,
            )
            self.evaluate(eval_steps=eval_steps)
        for hook in self.hooks:
            hook.end()

    def evaluate(self, eval_steps=None):
        """Execute the evaluation loop.

        Args:
            eval_steps (Optional[int]): Override for number of evaluation steps.
                                      If None, evaluates on full dataset.
        """
        self.restore()
        if hasattr(self.eval_dataset, "_window_paths"):
            iterator = self.get_new_window_iter(self.eval_dataset)
        else:
            iterator = iter(self.eval_dataset)
        lstep = 0
        while True:
            if eval_steps is not None and lstep >= eval_steps:
                break
            need_break = False
            stop_flag, data = next(iterator)
            if self.data_to_cuda:
                data = copy_data_to_device(data, "cuda")
            if stop_flag:
                if hasattr(self.eval_dataset, "_window_paths"):
                    iterator = self.get_new_window_iter(self.eval_dataset)
                    if iterator is None:
                        need_break = True
                else:
                    need_break = True
            need_break = self.sync_exit_flag(need_break)
            if need_break:
                break
            metrics = {}
            self.model.eval()
            with torch.no_grad():
                self.model(data)
            metrics.update(get_global_metrics())
            for hook in self.hooks:
                hook.after_step(metrics, self._global_step)
            lstep += 1
        for hook in self.hooks:
            hook.end()

    def get_new_window_iter(self, dataset):
        if not hasattr(dataset, "_window_paths"):
            raise TypeError("dataset must be window_io")
        while True:
            try:
                need_skip = dataset.next_window()
            except StopIteration:
                logger.info("Window IO Finish")
                return None
            except Exception as e:
                raise e
            read_offset = int(dataset._read_offset[0])
            if need_skip:
                logger.info(f"Skip for window, offset = {read_offset}")
            else:
                logger.info(f"Next window, offset = {read_offset}")
                break
        return iter(dataset)

    def sync_exit_flag(self, flag: bool):
        self.stop_state.fill_(int(flag))
        dist.all_reduce(self.stop_state, op=dist.ReduceOp.MAX)
        return bool(self.stop_state.item())

    def _train_loop(self, max_steps=None, epoch=1):
        self.model.train()
        if hasattr(self.train_dataset, "_window_paths"):
            iterator = self.get_new_window_iter(self.train_dataset)
        else:
            iterator = iter(self.train_dataset)
        lstep = 0
        while True:
            if max_steps is not None and lstep >= max_steps - 1:
                break
            stop_flag, data = next(iterator)
            if self.data_to_cuda:
                data = copy_data_to_device(data, "cuda")
            need_break = False
            if stop_flag:
                if hasattr(self.train_dataset, "_window_paths"):
                    iterator = self.get_new_window_iter(self.train_dataset)
                    if iterator is None:
                        need_break = True
                else:
                    need_break = True
            need_break = self.sync_exit_flag(need_break)
            if need_break:
                break
            metrics = {}
            with self.accelerator.accumulate(self.model):
                self._train_step(data, epoch, metrics)
            for hook in self.hooks:
                hook.after_step(metrics, self._global_step)
            self.checkpoint_manager.step()
            lstep += 1
        self.checkpoint_manager.save()

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
