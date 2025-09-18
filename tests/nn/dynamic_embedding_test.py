import os
import unittest

import torch
import torch.testing._internal.common_utils as common

from recis.nn.initializers import TruncNormalInitializer
from recis.nn.modules.embedding import DynamicEmbedding, EmbeddingOption
from recis.ragged.tensor import RaggedTensor


class DynamicEmbTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())
        os.environ["RANK"] = "0"
        torch.distributed.init_process_group()

    def test_dynamic_embedding(self):
        emb_opt = EmbeddingOption(
            embedding_dim=3,
            shared_name="ragged_ida",
            initializer=TruncNormalInitializer(mean=0, std=0.01),
        )
        if int(os.environ.get("RANK", 0)) == 0:
            ida = torch.tensor([5, 16385, 1, 32800, 16385], dtype=torch.int64).to(
                "cuda"
            )
            row_splits_a = torch.tensor([0, 1, 3, 5], dtype=torch.int32).to("cuda")
        else:
            ida = torch.tensor(
                [9155707040084860980, 11, 12, 20, 21, 30, 31, 32, 33], dtype=torch.int64
            ).to("cuda")
            row_splits_a = torch.tensor([0, 3, 5, 9], dtype=torch.int32).to("cuda")
        dea = DynamicEmbedding(emb_opt)
        rt = RaggedTensor(ida, row_splits_a)
        emba = dea(rt)
        target_emb = torch.randn(3, 3).to("cuda")
        loss = torch.mean((emba - target_emb) ** 2)
        loss.backward()

    @classmethod
    def tearDownClass(cls):
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
