import random
import unittest

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from recis.nn.functional.ragged_ops import feature_cross_ragged


def gen_2d_ragged_tensor(
    batch_size: int, max_row_length: int, device="cpu", dtype=torch.int64, seed=None
):
    if seed is not None:
        random.seed(seed)

    offsets = [0]
    all_values = []

    for i in range(batch_size):
        row_len = random.randint(1, max_row_length)
        if row_len > 0:
            # vals = torch.arange(row_len, dtype=dtype) + i * (max_row_length + 1)
            vals = torch.randint(low=0, high=9999, size=(row_len,))
            all_values.append(vals)
        offsets.append(offsets[-1] + row_len)

    assert len(all_values) > 0, "err when gen ragged!"

    values = torch.cat(all_values).to(device)
    # offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
    offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    return values, offsets


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
        self.max_row_length = 10
        self.num_tests = 5

    def test_cpu_gpu_consistency(self):
        for t in range(self.num_tests):
            seed = random.randrange(2**32)
            x_vals, x_offs = gen_2d_ragged_tensor(
                self.batch_size, self.max_row_length, seed=seed
            )
            y_vals, y_offs = gen_2d_ragged_tensor(
                self.batch_size, self.max_row_length, seed=seed + 1
            )
            print(f"len x vals {x_vals.shape}, x_offs {x_offs.shape}")
            print(f"len y vals {y_vals.shape}, y_offs {y_offs.shape}")
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


class TestFeatureCrossRaggedPerformance(unittest.TestCase):
    # @unittest.skip("Performance profiling not part of unit tests")
    def test_profile_timeline(self):
        bs, mrl, iterations = 3500, 1, 5
        # bs, mrl, iterations = 3500, 32, 2
        x_vals, x_offs = gen_2d_ragged_tensor(bs, mrl)
        y_vals, y_offs = gen_2d_ragged_tensor(bs, mrl)
        x_w = torch.rand_like(x_vals, dtype=torch.float32)
        y_w = torch.rand_like(y_vals, dtype=torch.float32)
        trace_file = f"feature_cross_ragged_{bs}_{mrl}_{iterations}.json"
        cpu_out_vals, cpu_out_offs, cpu_out_w = None, None, None
        gpu_out_vals, gpu_out_offs, gpu_out_w = None, None, None

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True,
            profile_memory=True,
            record_shapes=True,
            with_flops=True,
        ) as prof:
            with record_function("feature_cross_cpu"):
                for _ in range(iterations):
                    cpu_out_vals, cpu_out_offs, cpu_out_w = (
                        torch.ops.recis.feature_cross_ragged(
                            x_vals, x_offs, x_w, y_vals, y_offs, y_w
                        )
                    )
            with record_function("feature_cross_cuda"):
                xv_cuda = x_vals.cuda()
                xo_cuda = x_offs.cuda()
                xw_cuda = x_w.cuda()
                yv_cuda = y_vals.cuda()
                yo_cuda = y_offs.cuda()
                yw_cuda = y_w.cuda()
                for _ in range(iterations):
                    gpu_out_vals, gpu_out_offs, gpu_out_w = (
                        torch.ops.recis.feature_cross_ragged(
                            xv_cuda, xo_cuda, xw_cuda, yv_cuda, yo_cuda, yw_cuda
                        )
                    )
                torch.cuda.synchronize()
        prof.export_chrome_trace(trace_file)
        print(f"Profiling trace saved to {trace_file}")


if __name__ == "__main__":
    # unittest.main()
    suite = unittest.TestSuite()
    # suite.addTest(TestFeatureCrossRaggedPerformance("test_profile_timeline"))
    suite.addTest(TestFeatureCrossRaggedConsistency("test_cpu_gpu_consistency"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
