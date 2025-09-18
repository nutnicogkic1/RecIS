import unittest

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from recis.nn.functional.ragged_ops import fused_ragged_cutoff_2D


class TestFusedCutOffConsistency(unittest.TestCase):
    def setUp(self):
        values_a = torch.tensor(
            [9155707040084860980, 11, 12, 20, 21, 30, 31, 32, 33], dtype=torch.int64
        ).to("cuda")
        values_b = torch.tensor([7, 1, 2, 3, 4, 5, 6], dtype=torch.int64).to("cuda")
        values_c = torch.tensor([-7, -1, -2, -3, -4, -5, -6], dtype=torch.int64).to(
            "cuda"
        )

        row_splits_a = torch.tensor([0, 3, 5, 9], dtype=torch.int32).to("cuda")
        row_splits_b = torch.tensor([0, 1, 7], dtype=torch.int32).to("cuda")
        row_splits_c = torch.tensor([0, 1, 2, 4, 7], dtype=torch.int32).to("cuda")

        self.fused_vals = [values_a, values_b, values_c]
        self.fused_offsets = [row_splits_a, row_splits_b, row_splits_c]
        self.keep_lengths = torch.tensor([3, 4, 2], dtype=torch.int32).to("cuda")
        self.drop_sides = torch.tensor([False, False, True], dtype=torch.bool).to(
            "cuda"
        )
        self.pad_sides = torch.tensor([False, False, True], dtype=torch.bool).to("cuda")
        self.keep_lengths_list = self.keep_lengths.tolist()
        self.drop_sides_list = self.drop_sides.tolist()
        self.pad_sides_list = self.pad_sides.tolist()

        # self.load_data()   # use zhengxiong workload
        self.num_tests = 5
        self.trace_file = "merge_test.json"

    def test_profile_timeline(self):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True,
            profile_memory=True,
            record_shapes=True,
            with_flops=True,
        ) as prof:
            for t in range(self.num_tests):
                with record_function("fused_ragged_cutoff"):
                    fuse_res_tuple = fused_ragged_cutoff_2D(
                        self.fused_vals,
                        self.fused_offsets,
                        self.keep_lengths,
                        self.drop_sides,
                        self.pad_sides,
                    )
                list(
                    zip(
                        fuse_res_tuple[0],
                        fuse_res_tuple[1],
                        fuse_res_tuple[2],
                        fuse_res_tuple[3],
                    )
                )
                torch.cuda.synchronize()

        prof.export_chrome_trace(self.trace_file)
        print(f"Profile trace written to {self.trace_file}")


if __name__ == "__main__":
    # unittest.main()
    suite = unittest.TestSuite()
    # suite.addTest(TestFusedCutOffConsistency("test_fuse_consistency"))
    suite.addTest(TestFusedCutOffConsistency("test_profile_timeline"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
