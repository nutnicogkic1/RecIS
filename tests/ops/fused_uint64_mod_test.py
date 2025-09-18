import unittest

import numpy as np
import torch

from recis.nn.functional.fused_ops import fused_uint64_mod_gpu


def gen_data():
    shapes = [12, 13, 14, 15]
    ids_list = [
        torch.randint(0, 1000000, (shape,), dtype=torch.int64).cuda()
        for shape in shapes
    ]
    mod_list = torch.Tensor([2, 3, 4, 5]).to(dtype=torch.int64).cuda()
    return ids_list, mod_list


def uint64_mod_cpu(ids_list, mod_list):
    ret = []
    mod_list = mod_list.cpu().numpy()
    for i in range(len(ids_list)):
        data = ids_list[i].cpu().numpy()
        mod = data.astype(np.uint64) % mod_list[i]
        ret.append(torch.from_numpy(mod.astype(np.int64)))
    return ret


class FusedUInt64ModTest(unittest.TestCase):
    def test_fused_uint64_mod(self):
        ids_list, mod_list = gen_data()
        ans = fused_uint64_mod_gpu(ids_list, mod_list)
        ret = uint64_mod_cpu(ids_list, mod_list)
        for i in range(len(ans)):
            self.assertTrue(torch.equal(ans[i].cpu(), ret[i]))


if __name__ == "__main__":
    unittest.main()
