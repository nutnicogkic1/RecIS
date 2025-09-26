import random
import unittest
from collections import defaultdict

import torch

from recis.nn.functional.ragged_ops import fused_ragged_cutoff_2D


def gen_2d_ragged_tensor(
    bs: int,
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
    for i in range(bs):
        row_len = random.randint(min_seq, max_seq)
        max_row_len = max(max_row_len, row_len)
        value_len += row_len
        offsets.append(offsets[-1] + row_len)

    values = torch.arange(value_len, dtype=dtype, device=device)
    offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    return values, offsets, [max_row_len]


class TestFusedCutOffConsistency(unittest.TestCase):
    def setUp(self):
        self.mock_data()
        self.num_tests = 5

    def mock_data(self):
        self.fused_vals = defaultdict(list)
        self.fused_offsets = defaultdict(list)
        self.keep_lengths_list = defaultdict(list)
        self.keep_lengths = {}
        self.drop_sides_list = defaultdict(list)
        self.drop_sides = {}
        self.pad_sides_list = defaultdict(list)
        self.pad_sides = {}

        for i in range(5):
            val, offsets, max_lens = gen_2d_ragged_tensor(
                bs=3,
                min_seq=0,
                max_seq=0,
                dtype=torch.int64,
            )
            self.fused_vals[torch.int64].append(val.cuda())
            self.fused_offsets[torch.int64].append(offsets.cuda())
            self.keep_lengths_list[torch.int64].append(i + 3)
            self.drop_sides_list[torch.int64].append(False)
            self.pad_sides_list[torch.int64].append(False)
            # input:
            # value (tensor([], device='cuda:0', dtype=torch.int64)
            # offsets tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32).
            # keep_lengths 10
            # drop_sides False
            # pad_sides False
            # output:
            # value tensor([], device='cuda:0', dtype=torch.int64)
            # offsets tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0', dtype=torch.int32)
            # drop_num tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0', dtype=torch.int32)
            # pad_num tensor([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], device='cuda:0',dtype=torch.int32)),
        for i in range(20):
            val, offsets, max_lens = gen_2d_ragged_tensor(
                bs=4000,
                min_seq=60,
                max_seq=60 + i,
                dtype=torch.int64,
            )
            self.fused_vals[torch.int64].append(val.cuda())
            self.fused_offsets[torch.int64].append(offsets.cuda())
            self.keep_lengths_list[torch.int64].append(i % 2 + 20)
            self.drop_sides_list[torch.int64].append(False)
            self.pad_sides_list[torch.int64].append(False)
        for i in range(10):
            val, offsets, max_lens = gen_2d_ragged_tensor(
                bs=2000, min_seq=40, max_seq=40 + i, dtype=torch.int32
            )
            self.fused_vals[torch.int32].append(val.cuda())
            self.fused_offsets[torch.int32].append(offsets.cuda())
            self.keep_lengths_list[torch.int32].append(i % 2 + 20)
            self.drop_sides_list[torch.int32].append(False)
            self.pad_sides_list[torch.int32].append(False)
        for i in range(10):
            val, offsets, max_lens = gen_2d_ragged_tensor(
                bs=500,
                min_seq=0,
                max_seq=0,
                dtype=torch.float64,
            )
            self.fused_vals[torch.float64].append(val.cuda())
            self.fused_offsets[torch.float64].append(offsets.cuda())
            self.keep_lengths_list[torch.float64].append(i + 3)
            self.drop_sides_list[torch.float64].append(False)
            self.pad_sides_list[torch.float64].append(False)
        for i in range(10):
            val, offsets, max_lens = gen_2d_ragged_tensor(
                bs=2000,
                min_seq=1,
                max_seq=60 + i,
                dtype=torch.float32,
            )
            self.fused_vals[torch.float32].append(val.cuda())
            self.fused_offsets[torch.float32].append(offsets.cuda())
            self.keep_lengths_list[torch.float32].append(i % 8 + 20)
            self.drop_sides_list[torch.float32].append(False)
            self.pad_sides_list[torch.float32].append(False)
        for dt in [torch.float64, torch.int64, torch.int32, torch.float32]:
            if dt not in self.keep_lengths_list:
                continue
            self.keep_lengths[dt] = torch.tensor(
                self.keep_lengths_list[dt], dtype=torch.int32
            ).to("cuda")
            self.drop_sides[dt] = torch.tensor(
                self.drop_sides_list[dt], dtype=torch.bool
            ).to("cuda")
            self.pad_sides[dt] = torch.tensor(
                self.pad_sides_list[dt], dtype=torch.bool
            ).to("cuda")

    # Since recis removes the implementation of sparse cutoff, we compare calling fuse_cutoff_2D individually versus fusing the inputs and calling fuse_cutoff_2D once, in order to verify the correctness of the operator.
    def test_fuse_consistency(self):
        for t in range(self.num_tests):
            for dt in [torch.int64, torch.float64, torch.float32, torch.int32]:
                if dt not in self.fused_vals:
                    print(f"no type {dt}, continue")
                    continue
                fused_vals = self.fused_vals[dt]
                fused_offsets = self.fused_offsets[dt]
                keep_lengths_list = self.keep_lengths_list[dt]
                drop_sides_list = self.drop_sides_list[dt]
                pad_sides_list = self.pad_sides_list[dt]
                keep_lengths = self.keep_lengths[dt]

                res = []
                for idx, (val, offset, keep_len, drop, pad) in enumerate(
                    zip(
                        fused_vals,
                        fused_offsets,
                        keep_lengths_list,
                        drop_sides_list,
                        pad_sides_list,
                    )
                ):
                    tmp = fused_ragged_cutoff_2D(
                        [val],
                        [offset],
                        torch.tensor([keep_len], dtype=torch.int32).to("cuda"),
                        torch.tensor([drop]).to("cuda"),
                        torch.tensor([pad]).to("cuda"),
                    )
                    res.append((tmp[0][0], tmp[1][0], tmp[2][0], tmp[3][0]))
                    torch.cuda.synchronize()

                fuse_res_tuple = fused_ragged_cutoff_2D(
                    fused_vals,
                    fused_offsets,
                    keep_lengths,
                    self.drop_sides[dt],
                    self.pad_sides[dt],
                )
                torch.cuda.synchronize()
                fuse_res = list(
                    zip(
                        fuse_res_tuple[0],
                        fuse_res_tuple[1],
                        fuse_res_tuple[2],
                        fuse_res_tuple[3],
                    )
                )

                with self.subTest(test=t):
                    for idx, (t1, t2) in enumerate(zip(res, fuse_res)):
                        val, offset, drop, pad = t1
                        valf, offsetf, dropf, padf = t2
                        self.assertTrue(
                            torch.equal(val, valf),
                            f"val and valf Mismatch at test {val} and with {valf} seed",
                        )
                        self.assertTrue(
                            torch.equal(offset, offsetf),
                            f"offset and offsetf Mismatch at test {offset} and with {offsetf} seed",
                        )
                        self.assertTrue(
                            torch.equal(drop, dropf),
                            f"drop and dropf Mismatch at test {drop} and with {dropf} seed",
                        )
                        self.assertTrue(
                            torch.equal(pad, padf),
                            f"pad and padf Mismatch at test {pad} and with {padf} seed",
                        )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestFusedCutOffConsistency("test_fuse_consistency"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
