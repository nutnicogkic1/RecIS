import unittest

import torch

from recis.nn.functional.fused_ops import fused_bucketize_gpu


def gen_data():
    shapes = [12, 13, 14, 15]
    values = [
        torch.rand(shapes[i], device="cuda").float() * 10 for i in range(len(shapes))
    ]
    boundaries = [
        torch.Tensor([1, 2, 3, 4, 5]).float().cuda() for _ in range(len(shapes))
    ]
    return values, boundaries


class FusedUInt64ModTest(unittest.TestCase):
    def test_fused_uint64_mod(self):
        values, boundaries = gen_data()
        ans = fused_bucketize_gpu(values, boundaries)
        ret = [torch.bucketize(values[i], boundaries[i]) for i in range(4)]
        for i in range(len(ans)):
            self.assertTrue(torch.equal(ans[i].to(dtype=torch.int64), ret[i]))


if __name__ == "__main__":
    unittest.main()
