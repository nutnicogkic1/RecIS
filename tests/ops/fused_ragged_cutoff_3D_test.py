import random
import unittest
from collections import defaultdict
from typing import Optional

import torch

from recis.nn.functional.ragged_ops import _fused_ragged_cutoff_3D


def gen_ragged_tensor(
    bs: int,
    min_seq: int,
    max_seq: int,
    min_n: int,
    max_n: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
):
    if seed is not None:
        random.seed(seed)

    offsets = [[0], [0]]
    max_lengths = [0, 0]
    values_len = 0

    if max_seq < min_seq:
        raise ValueError("max_seq must be greater than or equal to min_seq.")
    if max_n < min_n:
        raise ValueError("max_n must be greater than or equal to min_n.")

    for i in range(bs):
        seq_len = random.randint(min_seq, max_seq)
        max_lengths[0] = max(max_lengths[0], seq_len)
        for j in range(seq_len):
            element_len = random.randint(min_n, max_n)
            values_len += element_len
            max_lengths[1] = max(max_lengths[1], element_len)
            offsets[-1].append(offsets[-1][-1] + element_len)

        offsets[0].append(offsets[0][-1] + seq_len)

    offsets = [torch.tensor(x, dtype=torch.int32, device=device) for x in offsets]
    values = torch.arange(values_len, dtype=dtype, device=device)

    return values, offsets, max_lengths


class TestFusedCutOffConsistency(unittest.TestCase):
    def setUp(self):
        # self.load_data()   # use real mainse workload
        self.mock_data()
        self.num_tests = 5
        self.trace_file = "mainse_3D_cutoff.json"

    def mock_data(self):
        self.fused_vals = defaultdict(list)
        self.fused_offsets = defaultdict(list)
        self.fused_inner_offsets = defaultdict(list)
        self.keep_lengths_list = defaultdict(list)
        self.keep_lengths = {}
        self.drop_sides_list = defaultdict(list)
        self.drop_sides = {}
        self.pad_sides_list = defaultdict(list)
        self.pad_sides = {}

        for i in range(10):
            val, offsets, max_lens = gen_ragged_tensor(
                bs=2000,
                min_seq=50,
                max_seq=50 + i,
                max_n=20 - i,
                min_n=5,
                dtype=torch.float64,
            )
            self.fused_vals[torch.float64].append(val.cuda())
            self.fused_offsets[torch.float64].append(offsets[0].cuda())
            self.fused_inner_offsets[torch.float64].append(offsets[1].cuda())
            self.keep_lengths_list[torch.float64].append(i + 10)
            self.drop_sides_list[torch.float64].append(False)
            self.pad_sides_list[torch.float64].append(False)

        for i in range(10):
            val, offsets, max_lens = gen_ragged_tensor(
                bs=2000,
                min_seq=60,
                max_seq=60 + i,
                max_n=10,
                min_n=1,
                dtype=torch.int64,
            )
            self.fused_vals[torch.int64].append(val.cuda())
            self.fused_offsets[torch.int64].append(offsets[0].cuda())
            self.fused_inner_offsets[torch.int64].append(offsets[1].cuda())
            self.keep_lengths_list[torch.int64].append(i % 2 + 20)
            self.drop_sides_list[torch.int64].append(False)
            self.pad_sides_list[torch.int64].append(False)

        for i in range(10):
            val, offsets, max_lens = gen_ragged_tensor(
                bs=2000, min_seq=2, max_seq=2 + i, max_n=1, min_n=0, dtype=torch.int32
            )
            self.fused_vals[torch.int32].append(val.cuda())
            self.fused_offsets[torch.int32].append(offsets[0].cuda())
            self.fused_inner_offsets[torch.int32].append(offsets[1].cuda())
            self.keep_lengths_list[torch.int32].append(i % 2 + 20)
            self.drop_sides_list[torch.int32].append(False)
            self.pad_sides_list[torch.int32].append(False)

        for i in range(10):
            val, offsets, max_lens = gen_ragged_tensor(
                bs=2000,
                min_seq=1,
                max_seq=10 + i,
                max_n=1,
                min_n=0,
                dtype=torch.float32,
            )
            self.fused_vals[torch.float32].append(val.cuda())
            self.fused_offsets[torch.float32].append(offsets[0].cuda())
            self.fused_inner_offsets[torch.float32].append(offsets[1].cuda())
            self.keep_lengths_list[torch.float32].append(i % 8 + 20)
            self.drop_sides_list[torch.float32].append(False)
            self.pad_sides_list[torch.float32].append(False)

        for dt in [torch.float64, torch.int64, torch.int32, torch.float32]:
            self.keep_lengths[dt] = torch.tensor(
                self.keep_lengths_list[dt], dtype=torch.int32
            ).to("cuda")
            self.drop_sides[dt] = torch.tensor(
                self.drop_sides_list[dt], dtype=torch.bool
            ).to("cuda")
            self.pad_sides[dt] = torch.tensor(
                self.pad_sides_list[dt], dtype=torch.bool
            ).to("cuda")

    # Since recis v2 removes the implementation of sparse cutoff, we compare calling fuse_cutoff_3D individually versus fusing the inputs and calling fuse_cutoff_3D once, in order to verify the correctness of the operator.
    def test_fuse_consistency(self):
        for t in range(self.num_tests):
            for dt in [torch.int64, torch.float64, torch.float32, torch.int32]:
                if dt not in self.fused_vals:
                    print(f"no type {dt}, continue")
                    continue
                fused_vals = self.fused_vals[dt]
                fused_offsets = self.fused_offsets[dt]
                fused_inner_offsets = self.fused_inner_offsets[dt]
                keep_lengths_list = self.keep_lengths_list[dt]
                drop_sides_list = self.drop_sides_list[dt]
                pad_sides_list = self.pad_sides_list[dt]
                keep_lengths = self.keep_lengths[dt]

                res = []
                for idx, (val, offset, inner_offsets, keep_len, drop, pad) in enumerate(
                    zip(
                        fused_vals,
                        fused_offsets,
                        fused_inner_offsets,
                        keep_lengths_list,
                        drop_sides_list,
                        pad_sides_list,
                    )
                ):
                    tmp = torch.ops.recis.fused_ragged_cutoff_3D(
                        [val],
                        [offset],
                        [inner_offsets],
                        torch.tensor([keep_len], dtype=torch.int32).to("cuda"),
                        torch.tensor([drop]).to("cuda"),
                        torch.tensor([pad]).to("cuda"),
                    )
                    res.append((tmp[0][0], tmp[1][0], tmp[2][0]))
                    torch.cuda.synchronize()

                fuse_res_tuple = _fused_ragged_cutoff_3D(
                    fused_vals, fused_offsets, fused_inner_offsets, keep_lengths
                )
                torch.cuda.synchronize()
                fuse_res = list(
                    zip(fuse_res_tuple[0], fuse_res_tuple[1], fuse_res_tuple[2])
                )

                with self.subTest(test=t):
                    for idx, (t1, t2) in enumerate(zip(res, fuse_res)):
                        val, outer_offset, inner_offset = t1
                        valf, outer_offsetf, inner_offsetf = t2
                        self.assertTrue(
                            torch.equal(outer_offset, outer_offsetf),
                            f"offset and offsetf Mismatch at test {outer_offset} and with {outer_offsetf} seed",
                        )
                        self.assertTrue(
                            torch.equal(inner_offset, inner_offsetf),
                            f"offset and offsetf Mismatch at test {inner_offset} and with {inner_offsetf} seed",
                        )
                        self.assertTrue(
                            torch.equal(val, valf),
                            f"val and valf Mismatch at test {val} and with {valf} seed",
                        )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestFusedCutOffConsistency("test_fuse_consistency"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
