import unittest

import torch

from recis.nn.functional.fused_ops import fused_ids_encode_gpu


def encode_ids(ids_list, offset_list):
    _MAX_BIT_SIZE = 64
    _MAX_ENCODE_SIZE = 12
    ret = []
    for ids, offset in zip(ids_list, offset_list):
        mask = torch.ones_like(ids, dtype=torch.int64, device=ids.device)
        mask = torch.bitwise_left_shift(mask, _MAX_BIT_SIZE - _MAX_ENCODE_SIZE) - 1
        offset = torch.bitwise_left_shift(offset, _MAX_BIT_SIZE - _MAX_ENCODE_SIZE)
        en_ids = torch.bitwise_and(ids, mask) + offset
        ret.append(en_ids)
    return torch.cat(ret, dim=0)


class IDEncodeTest(unittest.TestCase):
    def get_data(self, device="cpu"):
        shapes = [1000, 10000, 100000, 1000000]
        ids_list = [
            torch.randint(0, 1000000, (shape,), dtype=torch.int64, device=device)
            for shape in shapes
        ]
        offset_list = torch.arange(0, len(shapes), dtype=torch.int64, device=device)
        return ids_list, offset_list

    def test_ids_encode_cuda(self):
        ids_list, offset_list = self.get_data("cuda")
        ans = fused_ids_encode_gpu(ids_list, offset_list)
        ret = encode_ids(ids_list, offset_list)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(ans, ret))


if __name__ == "__main__":
    unittest.main()
