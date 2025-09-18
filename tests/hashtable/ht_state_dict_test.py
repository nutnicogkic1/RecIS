import unittest

import torch

from recis.nn.modules.hashtable import HashTable, filter_out_sparse_param
from recis.optim.sparse_adamw_tf import SparseAdamWTF


class HashTableTest(unittest.TestCase):
    @staticmethod
    def get_model():
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._emb_one = HashTable([16], name="16")
                self._emb_two = HashTable([32], name="32")
                self._emb_three = HashTable([64], name="64")

            def forward(self, ids) -> torch.Tensor:
                return torch.concat(
                    [self._emb_one(ids), self._emb_two(ids), self._emb_three(ids)],
                    dim=1,
                ).sum()

        return Model()

    def test_hash_table_dump(self):
        model = self.get_model()
        sparse_state = filter_out_sparse_param(model)
        optim = SparseAdamWTF(sparse_state)
        ids_num = (1 << 20) * 3 + 200
        ids = torch.arange(ids_num, device="cuda")
        for i in range(2):
            loss = model(ids)
            loss.backward()
            optim.step()
            optim.zero_grad()
        save_dict = {}
        save_dict.update(sparse_state)
        exp_keys = ["16", "32", "64"]
        self.assertTrue(list(save_dict.keys()) == exp_keys)
        exp_keys.extend(
            ["sparse_adamw_tf_beta2", "sparse_adamw_tf_beta1", "sparse_adamw_tf_step"]
        )
        save_dict.update(optim.state_dict())
        self.assertTrue(list(save_dict.keys()) == exp_keys)


if __name__ == "__main__":
    unittest.main()
