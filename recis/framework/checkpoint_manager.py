import os
from collections import OrderedDict
from typing import Optional

import torch

from recis.framework.filesystem import get_file_system
from recis.info import is_internal_enabled
from recis.nn.modules.hashtable import split_sparse_dense_state_dict
from recis.serialize import Loader as SLoader, Saver as SSaver
from recis.utils.logger import Logger


if is_internal_enabled():
    from pangudfs_client.common.exception.exceptions import PanguException

    from recis.utils.mos import Mos
else:
    PanguException = None
    Mos = None
logger = Logger(__name__)


class Saver:
    """Checkpoint saver for managing model and training state persistence.

    The Saver class handles the saving and loading of model checkpoints including:
    - Dense and sparse model parameters
    - Optimizer states
    - IO states for datasets
    - Checkpoint versioning and cleanup
    - Support for distributed filesystems

    Example:
        >>> saver = Saver(
        ...     model=model,
        ...     sparse_optim=sparse_optimizer,
        ...     output_dir="./checkpoints",
        ...     max_keep=5,
        ... )
        >>> saver.save("checkpoint_001")
    """

    kIndexSuffix = ".index"
    kIndexName = "index"

    def __init__(
        self,
        model: torch.nn.Module,
        sparse_optim=None,
        output_dir: str = "./",
        max_keep: int = 1,
        concurrency: int = 4,
    ):
        """Initialize the checkpoint saver.

        Args:
            model (torch.nn.Module): The model to save checkpoints for.
            sparse_optim (Optional): Sparse optimizer instance for sparse parameters.
            output_dir (str): Directory to save checkpoints. Defaults to "./".
            max_keep (int): Maximum number of checkpoints to keep. Defaults to 1.
            concurrency (int): Number of concurrent save operations. Defaults to 4.
        """
        self._model = model
        self._sparse_state_dict, self._dense_state_dict = split_sparse_dense_state_dict(
            model.state_dict()
        )
        self._checkpoint_file = "checkpoint"
        self._checkpoint_version_list = []
        self._max_keep = max_keep
        self._extra_save_dict = {}

        self._mos = None
        self._output_dir = output_dir
        if output_dir.startswith("model"):
            assert Mos is not None, "Cannot import mos, check interneal version."
            self._mos = Mos(output_dir)
            self._output_dir = self._mos.real_physical_path

        self._sparse_optim = sparse_optim
        self._sparse_optim_state = {}
        if sparse_optim is not None:
            self._sparse_optim_state = sparse_optim.state_dict()
            self._sparse_state_dict.update(self._sparse_optim_state)
        self._concurrency = concurrency
        self._io_state = {}

    def register_io_state(self, name, obj: object):
        """Register an object for IO state persistence.

        Args:
            name (str): Name identifier for the IO state.
            obj (object): Object that supports IO state dump/load operations.

        Raises:
            ValueError: If the name is already registered.
        """
        if name not in self._io_state:
            self._io_state[name] = obj
        else:
            raise ValueError(f"name {name} already registered in io state")

    def register_for_checkpointing(self, name, obj: object):
        """Register an object for checkpointing.

        Args:
            name (str): Name identifier for the checkpointed object.
            obj (object): Object to include in checkpoints.

        Raises:
            ValueError: If the name is already registered.
        """
        if name not in self._extra_save_dict:
            self._extra_save_dict[name] = obj
        else:
            raise ValueError(f"name {name} already registered")

    def save_sparse_meta(self, dirname: str):
        """Save sparse parameter metadata to index file.

        Args:
            dirname (str): Directory containing sparse parameter files.
        """
        fs = get_file_system(dirname)
        with fs.open(os.path.join(dirname, "index"), "w") as out_f:
            for filename in fs.listdir(dirname, detail=False):
                if filename.endswith(self.kIndexSuffix):
                    with fs.open(filename, "r") as inf:
                        out_f.write(inf.read())
                    fs.delete(filename)

    def save(self, ckpt_id: str, shard_id: int = 0, shard_num: int = 1):
        """Save a complete checkpoint with the given ID.

        This method saves all registered components including model parameters,
        optimizer states, and IO states. It also handles checkpoint versioning
        and cleanup of old checkpoints.

        Args:
            ckpt_id (str): Unique identifier for this checkpoint.
            shard_id (int): Shard ID for distributed saving. Defaults to 0.
            shard_num (int): Total number of shards. Defaults to 1.
        """
        ckpt_path = os.path.join(self._output_dir, ckpt_id)
        fs = get_file_system(ckpt_path)
        logger.info(f"Save checkpoint {ckpt_id} to {ckpt_path}")
        if not fs.exists(ckpt_path):
            try:
                fs.makedirs(ckpt_path + "/", exist_ok=True)
            except PanguException as e:
                if e.pangu_err_no == 7:
                    pass
        if len(self._sparse_state_dict.keys()) > 0:
            self.save_sparse_params(
                shard_id,
                shard_num,
                ckpt_path,
                self._sparse_state_dict,
                self._concurrency,
            )
        io_states = {}
        for io_name, io in self._io_state.items():
            io_states[io_name] = io.dump_io_state()
        if io_states:
            with fs.open(os.path.join(ckpt_path, f"io_state_{shard_id}.pt"), "wb") as f:
                torch.save(io_states, f=f)
        if shard_id == 0:
            if len(self._dense_state_dict.keys()) > 0:
                self.save_dense_params(ckpt_path, self._dense_state_dict)
            if len(self._extra_save_dict.keys()) > 0:
                extra_save = {}
                for key, value in self._extra_save_dict.items():
                    if hasattr(value, "state_dict"):
                        extra_save[key] = value.state_dict()
                    else:
                        extra_save[key] = value
                with fs.open(os.path.join(ckpt_path, "extra.pt"), "wb") as f:
                    torch.save(extra_save, f=f)
                if io_states:
                    with fs.open(os.path.join(ckpt_path, "io_state_count"), "w+") as f:
                        f.write(f"{shard_num}")
            with fs.open(
                os.path.join(self._output_dir, self._checkpoint_file), "a+"
            ) as out_f:
                out_f.write(ckpt_id + "\n")
                self._checkpoint_version_list.append(ckpt_id)
            if len(self._checkpoint_version_list) > self._max_keep:
                ckpt_id_to_remove = self._checkpoint_version_list[0]
                logger.info(
                    f"Remove checkpoint {os.path.join(self._output_dir, ckpt_id_to_remove)}"
                )
                fs.rm(
                    os.path.join(self._output_dir, ckpt_id_to_remove + "/"),
                    recursive=True,
                )
                remains = []
                with fs.open(
                    os.path.join(self._output_dir, self._checkpoint_file), "r"
                ) as f:
                    lines = [
                        line.strip()
                        for line in f.read().split("\n")
                        if len(line.strip()) != 0
                    ]
                    for ckpt_id in lines:
                        if ckpt_id != ckpt_id_to_remove:
                            remains.append(ckpt_id)
                with fs.open(
                    os.path.join(self._output_dir, self._checkpoint_file), "w"
                ) as f:
                    for ckpt_id in remains:
                        f.write(ckpt_id + "\n")
                self._checkpoint_version_list = self._checkpoint_version_list[1:]
                if self._mos:
                    self._mos.ckpt_update(
                        ckpt_id=ckpt_id_to_remove, path=ckpt_path, is_delete=True
                    )
            if self._mos:
                self._mos.ckpt_update(ckpt_id=ckpt_id, path=ckpt_path)
        torch.cuda.synchronize()

    def save_sparse_params(
        self,
        shard_id: int,
        shard_num: int,
        ckpt_path: str,
        sparse_state_dict: OrderedDict,
        concurrent: int = 16,
        sync_func=None,
    ):
        """Save sparse parameters using distributed saving.

        Args:
            shard_id (int): Current shard ID.
            shard_num (int): Total number of shards.
            ckpt_path (str): Path to save checkpoint.
            sparse_state_dict (OrderedDict): Sparse parameters to save.
            concurrent (int): Number of concurrent save operations. Defaults to 16.
            sync_func (Optional[Callable]): Synchronization function for distributed saving.
        """
        if not sync_func:
            if shard_num > 1:
                sync_func = torch.distributed.barrier
            else:

                def sync_func():
                    return None

        sparse_state_dict_copy = sparse_state_dict.copy()
        sparse_state_dict, dense_state_dict = split_sparse_dense_state_dict(
            sparse_state_dict_copy
        )
        saver = SSaver(
            shard_index=shard_id,
            shard_num=shard_num,
            parallel=concurrent,
            hashtables=sparse_state_dict,
            tensors=dense_state_dict,
            path=ckpt_path,
        )
        saver.save()
        sync_func()

    def save_dense_params(self, ckpt_path: str, dense_state_dict: OrderedDict):
        """Save dense model parameters.

        Args:
            ckpt_path (str): Path to save checkpoint.
            dense_state_dict (OrderedDict): Dense parameters to save.
        """
        fs = get_file_system(ckpt_path)
        pt_file = os.path.join(ckpt_path, "model.pt")
        with fs.open(pt_file, "wb") as f:
            torch.save(dense_state_dict, f=f)

    def load_sparse_params(self, ckpt_dir: str):
        """Load sparse parameters from checkpoint.

        Args:
            ckpt_dir (str): Directory containing the checkpoint.
        """
        sparse_state_copy = self._sparse_state_dict.copy()
        sparse_state_dict, dense_state_dict = split_sparse_dense_state_dict(
            sparse_state_copy
        )
        loader = SLoader(
            ckpt_dir, hashtables=sparse_state_dict, tensors=dense_state_dict
        )
        loader.load()

    def load_dense_params(self, ckpt_dir: str, strict: bool = True):
        """Load dense model parameters from checkpoint.

        Args:
            ckpt_dir (str): Directory containing the checkpoint.
            strict (bool): Whether to strictly enforce state dict keys match. Defaults to True.
        """
        logger.info("Load dense model")
        pt_file = os.path.join(ckpt_dir, "model.pt")
        fs = get_file_system(ckpt_dir)
        with fs.open(pt_file, "rb") as f:
            self._model.load_state_dict(torch.load(f=f), strict=strict)

    def load_extra_params(self, ckpt_dir: str, load_io: bool, shared_id: int = 0):
        """Load extra parameters and IO states from checkpoint.

        Args:
            ckpt_dir (str): Directory containing the checkpoint.
            load_io (bool): Whether to load IO states.
            shared_id (int): Shard ID for loading IO states. Defaults to 0.
        """
        extra_file = os.path.join(ckpt_dir, "extra.pt")
        fs = get_file_system(extra_file)
        with fs.open(extra_file, "rb") as f:
            extra_data = torch.load(f=f)
        if load_io:
            with fs.open(os.path.join(ckpt_dir, "io_state_count"), "r") as f:
                shard_num = int(f.read())
            with fs.open(os.path.join(ckpt_dir, f"io_state_{shared_id}.pt"), "rb") as f:
                io_state = torch.load(f=f)
            for io_name, io in self._io_state.items():
                assert shard_num == io._worker_num, (
                    f"IO states size not equal to worker num, expect: {io._worker_num}, got: {shard_num}"
                )
                if io_name in io_state:
                    logger.info(f"Load io state for dataset: {io_name}")
                    io.load_io_state(io_state[io_name])
                else:
                    logger.info(f"No io state found for dataset: {io_name}")
        else:
            logger.info("No need to load io state")
        for key, value in self._extra_save_dict.items():
            if hasattr(value, "load_state_dict"):
                value.load_state_dict(extra_data[key])
            else:
                value = extra_data[key]
            self._extra_save_dict[key] = value

    def load_sparse_optim(self):
        """Load sparse optimizer state from checkpoint."""
        if self._sparse_optim:
            logger.info("Load sparse optim")
            self._sparse_optim.load_state_dict(self._sparse_optim_state)

    def load(
        self,
        ckpt_path: Optional[str] = None,
        ckpt_id: Optional[str] = None,
        load_conf: Optional[dict] = None,
        shard_id: int = 0,
        direct_path=False,
    ):
        if load_conf is None:
            load_conf = {}
        if direct_path:
            ckpt_path = ckpt_path
            if not ckpt_path:
                return
            logger.info(f"Load checkpoint conf {load_conf} from {ckpt_path}")
        else:
            ckpt_path = self._output_dir if not ckpt_path else ckpt_path
            fs = get_file_system(ckpt_path)
            if ckpt_id is None:
                if fs.exists(os.path.join(ckpt_path, self._checkpoint_file)):
                    content = fs.open(
                        os.path.join(ckpt_path, self._checkpoint_file), "r"
                    ).read()
                    lines = content.split("\n")[::-1]
                    ckpt_id = None
                    for line in lines:
                        if len(line) == 0:
                            continue
                        ckpt_id = line.strip()
                        break
                else:
                    logger.info(f"Checkpoint not found in {ckpt_path}")
                    return
            logger.info(f"Load checkpoint conf {load_conf} from {ckpt_path}")
            ckpt_path = os.path.join(ckpt_path, ckpt_id)
        self.load_by_config(ckpt_path, load_conf, shard_id)

    def load_by_config(
        self, ckpt_path: str, load_conf: Optional[dict] = None, shared_id: int = 0
    ):
        # TODO update real load for model bank
        if load_conf is None:
            load_conf = {}
        if not load_conf:
            load_conf = {}
        load_map = {
            "sparse": False,
            "dense": False,
            "extra": False,
            "sparse_opt": False,
            "io_state": False,
        }
        for load in load_conf.get("load", ["*"]):
            if load == "*":
                load_map["sparse"] = True
                load_map["dense"] = True
                load_map["extra"] = True
                load_map["sparse_opt"] = True
                load_map["io_state"] = True
            else:
                load_map[load] = True
        for exclude in load_conf.get("exclude", []):
            if exclude == "*":
                load_map["sparse"] = False
                load_map["dense"] = False
                load_map["extra"] = False
                load_map["sparse_opt"] = False
                load_map["io_state"] = False
            else:
                load_map[exclude] = False
        strict = load_conf.get("strict", True)
        if load_map["sparse"] and len(self._sparse_state_dict.keys()) > 0:
            self.load_sparse_params(ckpt_path)
        if load_map["dense"] and len(self._dense_state_dict.keys()) > 0:
            self.load_dense_params(ckpt_path, strict=strict)
        if load_map["extra"] and len(self._extra_save_dict.keys()) > 0:
            self.load_extra_params(ckpt_path, load_map["io_state"], shared_id)
        if load_map["sparse_opt"] and len(self._sparse_optim_state.keys()) > 0:
            self.load_sparse_optim()

    def get_extra_data(self, name: str):
        if name in self._extra_save_dict:
            return self._extra_save_dict[name]
        else:
            return None


class CheckpointManager:
    """High-level checkpoint manager for coordinating checkpoint operations.

    The CheckpointManager provides a high-level interface for managing checkpoints
    during training, including automatic saving at intervals, loading from model banks,
    and coordinating with the training loop.

    Example:
        >>> checkpoint_manager = CheckpointManager(saver=saver, save_interval=1000)
        >>> # During training loop
        >>> checkpoint_manager.step()  # Call after each training step
        >>> # Automatic save will occur every save_interval steps
    """

    def __init__(self, saver: Saver, save_interval: int) -> None:
        """Initialize the checkpoint manager.

        Args:
            saver (Saver): The saver instance to use for checkpoint operations.
            save_interval (int): Number of steps between automatic saves.
        """
        self._saver = saver
        self._global_step = torch.scalar_tensor(0, dtype=torch.int64)
        self._rank = int(os.environ.get("RANK", 0))
        self._shard_num = int(os.environ.get("WORLD_SIZE", 1))
        self._save_interval = save_interval
        if not self._saver.get_extra_data("global_step"):
            self._saver.register_for_checkpointing("global_step", self._global_step)

    @property
    def save_interval(self):
        return self._save_interval

    def register_for_checkpointing(self, name, obj: object):
        self._saver.register_for_checkpointing(name, obj)

    def step(self):
        """Increment step counter and save checkpoint if interval is reached.

        This method should be called after each training step. It automatically
        saves a checkpoint when the step count reaches the save interval.
        """
        self._global_step += 1
        if self._global_step % self._save_interval == 0:
            ckpt_id = f"ckpt_{self._global_step}"
            self._saver.save(ckpt_id, self._rank, self._shard_num)

    def save(self):
        """Save a checkpoint with automatic ID generation."""
        ckpt_id = f"ckpt_{self._global_step}"
        self._saver.save(ckpt_id, self._rank, self._shard_num)

    def load_model_bank(self, model_bank_conf: Optional[dict]):
        if not model_bank_conf:
            return
        for mbc in model_bank_conf:
            path = mbc["path"]
            if path is not None and path.startswith("model."):
                assert Mos is not None, "Cannot import mos, check interneal version."
                path = Mos(path, True).real_physical_path
            self._saver.load(
                ckpt_path=path, load_conf=mbc, shard_id=self._rank, direct_path=True
            )

    def restore(self, global_step: Optional[int] = None):
        if global_step is None:
            ckpt_id = None
        else:
            ckpt_id = f"ckpt_{global_step}"
        self._saver.load(ckpt_id=ckpt_id, shard_id=self._rank)
        global_step = self._saver.get_extra_data("global_step")
        if global_step is not None:
            self._global_step = global_step
        return self._global_step
