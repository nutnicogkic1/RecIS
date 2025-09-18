import unittest

import torch

from recis.nn.functional.ragged_ops import ragged_topk_index_cutoff


class TestRagged(unittest.TestCase):
    def test_ragged_topk_index_cutoff(self):
        offset = torch.tensor([0, 8, 10, 13], dtype=torch.int32)
        drop_num = torch.tensor([5, 0, 0])
        pad_num = torch.tensor([0, 0, 0])
        drop_side = torch.tensor(True)
        pad_side = torch.tensor(False)
        topk_index = torch.tensor([[1, 2], [0, 2], [0, 1]])
        indicator = torch.tensor([0, 1, 2])
        value_index, offset = ragged_topk_index_cutoff(
            drop_num, pad_num, drop_side, pad_side, offset, topk_index, indicator
        )
        print(value_index, offset)
        self.assertTrue(torch.equal(value_index, torch.tensor([6, 7, 8, 10, 11])))
        self.assertTrue(torch.equal(offset, torch.tensor([0, 2, 3, 5])))


if __name__ == "__main__":
    unittest.main()
