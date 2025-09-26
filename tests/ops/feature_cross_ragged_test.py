import random
import unittest

import torch

from recis.nn.functional.ragged_ops import feature_cross_ragged


def gen_2d_ragged_tensor(
    batch_size: int,
    min_seq: int,
    max_seq: int,
    device="cpu",
    dtype=torch.int64,
    seed=None,
):
    if seed is not None:
        random.seed(seed)

    offsets = [0]
    value_len = 0

    max_row_len = 0
    for i in range(batch_size):
        row_len = random.randint(min_seq, max_seq)
        max_row_len = max(max_row_len, row_len)
        value_len += row_len
        offsets.append(offsets[-1] + row_len)

    values = torch.arange(value_len, dtype=dtype, device=device)
    offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    return values, offsets, [max_row_len]


def ragged_equal_rows(
    values1: torch.Tensor,
    offsets1: torch.Tensor,
    values2: torch.Tensor,
    offsets2: torch.Tensor,
) -> bool:
    if offsets1.numel() != offsets2.numel():
        return False
    batch_size = offsets1.numel() - 1
    for i in range(batch_size):
        s1, e1 = offsets1[i].item(), offsets1[i + 1].item()
        s2, e2 = offsets2[i].item(), offsets2[i + 1].item()
        if (e1 - s1) != (e2 - s2):
            return False
        row1 = values1[s1:e1]
        row2 = values2[s2:e2]
        u1, c1 = torch.unique(row1, return_counts=True)
        u2, c2 = torch.unique(row2, return_counts=True)
        if sorted(zip(u1.tolist(), c1.tolist())) != sorted(
            zip(u2.tolist(), c2.tolist())
        ):
            return False
    return True


class TestFeatureCrossRaggedConsistency(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2000
        self.min_seq = 1
        self.max_seq = 5
        self.num_tests = 8

    def test_cpu_gpu_consistency(self):
        for t in range(self.num_tests):
            seed = random.randrange(2**32)
            if t % 3 == 0:
                x_vals, x_offs, _ = gen_2d_ragged_tensor(
                    0, self.min_seq, self.max_seq, seed=seed
                )
                y_vals, y_offs, _ = gen_2d_ragged_tensor(
                    self.batch_size, self.min_seq, self.max_seq, seed=seed + 1
                )
            elif t % 3 == 1:
                x_vals, x_offs, _ = gen_2d_ragged_tensor(
                    self.batch_size, 0, 0, seed=seed
                )
                y_vals, y_offs, _ = gen_2d_ragged_tensor(
                    self.batch_size, self.min_seq, self.max_seq, seed=seed + 1
                )
            else:
                x_vals, x_offs, _ = gen_2d_ragged_tensor(
                    self.batch_size + 1, self.min_seq, self.max_seq, seed=seed
                )
                y_vals, y_offs, _ = gen_2d_ragged_tensor(
                    self.batch_size, self.min_seq, self.max_seq, seed=seed + 1
                )

            x_w = torch.rand_like(x_vals, dtype=torch.float32)
            y_w = torch.rand_like(y_vals, dtype=torch.float32)

            # CPU
            cpu_out_vals, cpu_out_offs, cpu_out_w = feature_cross_ragged(
                x_vals, x_offs, y_vals, y_offs, x_w, y_w
            )
            # GPU
            xv_cuda = x_vals.cuda()
            xo_cuda = x_offs.cuda()
            xw_cuda = x_w.cuda()
            yv_cuda = y_vals.cuda()
            yo_cuda = y_offs.cuda()
            yw_cuda = y_w.cuda()
            gpu_out_vals, gpu_out_offs, gpu_out_w = feature_cross_ragged(
                xv_cuda, xo_cuda, yv_cuda, yo_cuda, xw_cuda, yw_cuda
            )
            torch.cuda.synchronize()
            with self.subTest(test=t, seed=seed):
                self.assertTrue(
                    torch.equal(cpu_out_offs.cuda(), gpu_out_offs),
                    f"Offsets differ at test {t}, seed {seed}, cpu {cpu_out_offs}, gpu {gpu_out_offs.cpu()}",
                )
                self.assertTrue(
                    ragged_equal_rows(
                        cpu_out_vals,
                        cpu_out_offs,
                        gpu_out_vals.cpu(),
                        gpu_out_offs.cpu(),
                    ),
                    f"Values differ at test {t}, seed {seed}",
                )
                self.assertTrue(
                    ragged_equal_rows(
                        cpu_out_w, cpu_out_offs, gpu_out_w.cpu(), gpu_out_offs.cpu()
                    ),
                    f"Weights differ at test {t}, seed {seed}",
                )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestFeatureCrossRaggedConsistency("test_cpu_gpu_consistency"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
