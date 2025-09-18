import os
import unittest

import torch
import torch.testing._internal.common_utils as common

from recis.nn.initializers import TruncNormalInitializer
from recis.nn.modules.embedding import EmbeddingOption
from recis.nn.modules.embedding_engine import EmbeddingEngine
from recis.ragged.tensor import RaggedTensor


class EmbeddingEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())
        os.environ["RANK"] = "0"
        torch.distributed.init_process_group()

    def test_embedding_engine(self):
        id1 = torch.tensor(
            [5, 16385, 1, 32800, 16385], dtype=torch.int64, device="cuda"
        )
        row_splits_1 = torch.tensor([0, 1, 3, 5], dtype=torch.int64, device="cuda")
        rt1 = RaggedTensor(id1, row_splits_1)
        emb_opt1 = EmbeddingOption(
            embedding_dim=8,
            shared_name="ht2",
            combiner="sum",
            initializer=TruncNormalInitializer(mean=0, std=0.01),
        )

        id2 = torch.tensor(
            [9155707040084860980, 11, 12, 20, 21, 30, 31, 32, 33],
            dtype=torch.int64,
            device="cuda",
        )
        row_splits_2 = torch.tensor([0, 3, 5, 9], dtype=torch.int64, device="cuda")
        rt2 = RaggedTensor(id2, row_splits_2)
        emb_opt2 = EmbeddingOption(
            embedding_dim=8,
            shared_name="ht1",
            combiner="mean",
            initializer=TruncNormalInitializer(mean=0, std=0.01),
        )
        ee = EmbeddingEngine({"fea1": emb_opt1, "fea2": emb_opt2})
        out = ee({"fea1": rt1, "fea2": rt2})
        print(out)

    @classmethod
    def tearDownClass(cls):
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
