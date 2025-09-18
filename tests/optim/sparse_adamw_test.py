import unittest

import numpy as np
import torch

from recis.nn.modules.hashtable import HashTable, filter_out_sparse_param
from recis.optim.sparse_adamw_tf import SparseAdamWTF


def method_decorator(func):
    def wrapper(self, *args, **kargs):
        print(f"call test_func: {func.__name__}")
        return func(self, *args, **kargs)

    return wrapper


class Test(unittest.TestCase):
    @staticmethod
    def get_model(emb_size=4, order=3, name="test"):
        class Model(torch.nn.Module):
            def __init__(self, emb_size, order) -> None:
                super().__init__()
                self._emb_one = HashTable(
                    [emb_size], block_size=1024, device=torch.device("cuda"), name=name
                )
                self._dense_param = torch.nn.Parameter(
                    torch.zeros([order, emb_size], dtype=torch.float32)
                )
                self._order = order
                self._cur_order = 0

            def forward(self, ids):
                ids = ids.cuda()
                if self._cur_order >= self._order:
                    raise ValueError(f"call time exceed {self._order}")
                ret = torch.concat(
                    [self._emb_one(ids), self._dense_param[: self._cur_order + 1, :]],
                    dim=0,
                )
                self._cur_order += 1
                return ret

            @property
            def order(self):
                return self._cur_order

        return Model(emb_size, order)

    @staticmethod
    def distance(lhv: torch.Tensor, rhv: torch.Tensor):
        return torch.abs(lhv - rhv)

    @method_decorator
    def test_sparse_adamw_runnable(self):
        # no coredump is successs
        model = self.get_model(name="test1")
        model = model.cuda()
        sparse_param = filter_out_sparse_param(model)
        optim = SparseAdamWTF(param_dict=sparse_param)
        ids = torch.arange(100)
        emb = model(ids)
        loss = torch.sum(emb)
        loss.backward()
        optim.step()

    @method_decorator
    def test_sparse_adamw(self):
        model = self.get_model(name="test2")
        model = model.cuda()
        sparse_param = filter_out_sparse_param(model)
        optim = SparseAdamWTF(param_dict=sparse_param)
        dense_optim = torch.optim.AdamW(model.parameters())
        ids = torch.arange(3)
        emb = model(ids)
        loss = torch.sum(emb)
        loss.backward()
        optim.step()
        dense_optim.step()
        optim.zero_grad()
        dense_optim.zero_grad()

    @method_decorator
    def test_sparse_adamw_train_missing(self):
        model = self.get_model(name="test3")
        model = model.cuda()
        sparse_param = filter_out_sparse_param(model)
        optim = SparseAdamWTF(param_dict=sparse_param)
        dense_optim = torch.optim.AdamW(model.parameters())
        # not full block
        not_full_ids_num = 512
        ids = torch.arange(not_full_ids_num)
        emb = model(ids)
        loss = torch.sum(emb)
        loss.backward()
        optim.step()
        dense_optim.step()
        emb_one_embs = model._emb_one.embeddings()
        dense_emb = model._dense_param
        self.assertTrue(
            np.allclose(
                emb_one_embs[:not_full_ids_num].detach().cpu(),
                dense_emb[0, :].detach().cpu(),
                1e-5,
                1e-5,
            )
        )
        optim.zero_grad()
        dense_optim.zero_grad()
        # full block
        full_ids_num = 1024
        ids = torch.arange(full_ids_num)
        emb = model(ids)
        loss = torch.sum(emb)
        loss.backward()
        optim.step()
        dense_optim.step()
        emb_one_embs = model._emb_one.embeddings()
        dense_emb = model._dense_param
        dense_emb_one = torch.tensor([list(dense_emb[0])])
        self.assertTrue(
            np.allclose(
                emb_one_embs[:not_full_ids_num].detach().cpu(),
                dense_emb[0, :].detach().cpu(),
                1e-5,
                1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                emb_one_embs[:not_full_ids_num].detach().cpu(),
                dense_emb_one.detach().cpu(),
                1e-5,
                1e-5,
            )
        )
        optim.zero_grad()
        dense_optim.zero_grad()
        # more than one block
        over_ids_num = 1024 + 512
        ids = torch.arange(over_ids_num)
        emb = model(ids)
        loss = torch.sum(emb)
        loss.backward()
        optim.step()
        dense_optim.step()
        emb_one_embs = model._emb_one.embeddings()
        dense_emb = model._dense_param
        self.assertTrue(
            np.allclose(
                emb_one_embs[:not_full_ids_num].detach().cpu(),
                dense_emb[0, :].detach().cpu(),
                1e-4,
                1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                emb_one_embs[not_full_ids_num:full_ids_num].detach().cpu(),
                dense_emb[1, :].detach().cpu(),
                1e-4,
                1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                emb_one_embs[full_ids_num:over_ids_num].detach().cpu(),
                dense_emb[2, :].detach().cpu(),
                1e-4,
                1e-4,
            )
        )
        optim.zero_grad()
        dense_optim.zero_grad()

    @method_decorator
    def test_sparse_adamw_train_no_missing(self):
        model = self.get_model(name="test4")
        model = model.cuda()
        sparse_param = filter_out_sparse_param(model)
        optim = SparseAdamWTF(param_dict=sparse_param)
        model._emb_one.insert(
            torch.arange(2048), torch.zeros([2048, 4], dtype=torch.float32)
        )
        dense_optim = torch.optim.AdamW(model.parameters())
        # not full block
        not_full_ids_num = 512
        ids = torch.arange(not_full_ids_num)
        emb = model(ids)
        loss = torch.sum(emb)
        loss.backward()
        optim.step()
        dense_optim.step()
        emb_one_embs = model._emb_one.embeddings()
        dense_emb = model._dense_param
        self.assertTrue(
            np.allclose(
                emb_one_embs[2048 - not_full_ids_num : 2048].detach().cpu(),
                dense_emb[0, :].detach().cpu(),
                1e-5,
                1e-5,
            )
        )
        self.assertTrue(
            np.allclose(
                emb_one_embs[: 2048 - not_full_ids_num].detach().cpu(), 0.0, 1e-5, 1e-5
            )
        )
        optim.zero_grad()
        dense_optim.zero_grad()
        # full block
        full_ids_num = 1024
        ids = torch.arange(full_ids_num)
        emb = model(ids)
        loss = torch.sum(emb)
        loss.backward()
        optim.step()
        dense_optim.step()
        emb_one_embs = model._emb_one.embeddings()
        dense_emb = model._dense_param
        self.assertTrue(
            np.allclose(
                emb_one_embs[2048 - not_full_ids_num : 2048].detach().cpu(),
                dense_emb[0, :].detach().cpu(),
                1e-5,
                1e-5,
            )
        )
        self.assertTrue(
            np.allclose(
                emb_one_embs[2048 - full_ids_num : 2048 - not_full_ids_num]
                .detach()
                .cpu(),
                dense_emb[1, :].detach().cpu(),
                1e-5,
                1e-5,
            )
        )
        self.assertTrue(
            np.allclose(
                emb_one_embs[: 2048 - full_ids_num].detach().cpu(), 0.0, 1e-7, 1e-7
            )
        )
        optim.zero_grad()
        dense_optim.zero_grad()
        # more than one block
        over_ids_num = 1024 + 512
        ids = torch.arange(over_ids_num)
        emb = model(ids)
        loss = torch.sum(emb)
        loss.backward()
        optim.step()
        dense_optim.step()
        emb_one_embs = model._emb_one.embeddings()
        dense_emb = model._dense_param
        self.assertTrue(
            np.allclose(
                emb_one_embs[2048 - not_full_ids_num : 2048].detach().cpu(),
                dense_emb[0, :].detach().cpu(),
                1e-4,
                1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                emb_one_embs[2048 - full_ids_num : 2048 - not_full_ids_num]
                .detach()
                .cpu(),
                dense_emb[1, :].detach().cpu(),
                1e-4,
                1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                emb_one_embs[2048 - over_ids_num : 2048 - full_ids_num].detach().cpu(),
                dense_emb[2, :].detach().cpu(),
                1e-4,
                1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                emb_one_embs[: 2048 - over_ids_num].detach().cpu(), 0.0, 1e-7, 1e-7
            )
        )
        optim.zero_grad()
        dense_optim.zero_grad()

    @method_decorator
    def test_sparse_adamw_eval(self):
        model = self.get_model(name="test5")
        model = model.cuda()
        model.eval()
        sparse_param = filter_out_sparse_param(model)
        optim = SparseAdamWTF(param_dict=sparse_param)
        dense_optim = torch.optim.AdamW(model.parameters())
        # not full block
        not_full_ids_num = 512
        ids = torch.arange(not_full_ids_num)
        emb = model(ids)
        loss = torch.sum(emb)
        loss.backward()
        optim.step()
        dense_optim.step()
        emb_one_embs = model._emb_one.embeddings()
        self.assertTrue(emb_one_embs.numel() == 1024 * 4)
        optim.zero_grad()
        dense_optim.zero_grad()
        # full block
        full_ids_num = 1024
        ids = torch.arange(full_ids_num)
        emb = model(ids)
        loss = torch.sum(emb)
        loss.backward()
        optim.step()
        dense_optim.step()
        emb_one_embs = model._emb_one.embeddings()
        self.assertTrue(emb_one_embs.numel() == 1024 * 4)
        optim.zero_grad()
        dense_optim.zero_grad()
        # more than one block
        over_ids_num = 1024 + 512
        ids = torch.arange(over_ids_num)
        emb = model(ids)
        loss = torch.sum(emb)
        loss.backward()
        optim.step()
        dense_optim.step()
        torch.logical_and
        emb_one_embs = model._emb_one.embeddings()
        self.assertTrue(emb_one_embs.numel() == 1024 * 4)
        optim.zero_grad()
        dense_optim.zero_grad()


if __name__ == "__main__":
    unittest.main()
