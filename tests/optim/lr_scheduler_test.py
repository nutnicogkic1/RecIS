import os
import unittest

import torch
import torch.testing._internal.common_utils as common
from torch import nn

from recis.nn.modules.embedding import DynamicEmbedding, EmbeddingOption
from recis.nn.modules.hashtable import filter_out_sparse_param
from recis.optim.sparse_adamw_tf import SparseAdamWTF


class LRSchedulerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())
        os.environ["RANK"] = "0"
        torch.distributed.init_process_group()

    @staticmethod
    def get_model():
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                emb_opt = EmbeddingOption(
                    embedding_dim=16,
                    shared_name="test",
                )
                self.hasthtable = DynamicEmbedding(emb_opt)
                self.linear = nn.Linear(16, 128)

            def forward(self, ids):
                x = self.hasthtable(ids)
                x = self.linear(x)
                loss = x.mean()
                return loss

        return TestModel()

    def setUp(self):
        self.model = self.get_model()
        self.model.cuda()
        self.model.train()
        sparse_param = filter_out_sparse_param(self.model)
        self.dense_opt = torch.optim.AdamW(self.model.parameters(), lr=0.1)
        self.sparse_opt = SparseAdamWTF(sparse_param, lr=0.1)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.sparse_opt, step_size=10, gamma=0.95
        )

    def test_lr(self):
        for _ in range(100):
            ids = torch.randint(0, 2**52, [10, 100]).cuda()
            loss = self.model(ids)
            loss.backward()
            self.lr_scheduler.step()
            print(f"lr = {self.lr_scheduler.get_lr()}")
            self.dense_opt.step()
            self.sparse_opt.step()
            self.dense_opt.zero_grad()
            self.sparse_opt.zero_grad()

    @classmethod
    def tearDownClass(cls):
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
