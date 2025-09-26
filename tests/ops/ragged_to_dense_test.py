import random
import unittest
from typing import List, Optional

import torch

from recis.nn.functional.ragged_ops import ragged_to_dense


def ragged_to_dense_torch(
    values: torch.Tensor, offsets: List[torch.Tensor], default_value=0
):
    """
    Converts this `RaggedTensor` into a `tf.Tensor`, support nested dense_value and multi-dim offsets
    Example:

    .. code-block:: python

        rtd = ragged_to_dense(
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12]),
            [torch.tensor([0, 2, 2, 3, 6]), torch.tensor([0, 2, 4, 6, 9, 11, 13])],
            default_value=0,
        )
        print(rtd)
        [
            [[1, 2, 0], [3, 4, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[5, 6, 0], [0, 0, 0], [0, 0, 0]],
            [[7, 8, 9], [9, 10, 0], [11, 12, 0]],
        ]

    Args:
        default_value: Value to set for indices not specified in `self`. Defaults to zero
    Returns:
        torch.Tensor
    """
    device = values.device
    cur_offsets = offsets[0]
    inner_offsets = offsets[1:]
    # Recursively for multi-dim offsets
    if inner_offsets:
        # Check if offsets valid
        assert cur_offsets[-1] == len(inner_offsets[0]) - 1, (
            f"L layer offsets[-1] {cur_offsets[-1]} should equal L+1 layer offsets cnt {len(inner_offsets[0])}"
        )
        inner_values = ragged_to_dense_torch(values, inner_offsets, default_value)
    else:
        inner_values = values
    # Get the expected dense shape ([nrows, ncols] + value_shape
    nrows = cur_offsets.shape[0] - 1
    rt_row_lengths = cur_offsets[1:] - cur_offsets[:-1]
    ncols = rt_row_lengths.max().item() if nrows > 0 else 0
    inner_shape = inner_values.shape[1:]
    output_shape = torch.Size([nrows, ncols] + list(inner_shape))
    # Get the output tensor
    output = torch.full(output_shape, default_value, dtype=values.dtype, device=device)

    # Get the row start indices, and expand to shape=[nrows, 1].
    starts = cur_offsets[:-1].unsqueeze(1)  # 形状 (nrows, 1)
    # Get the column indices, and expand to shape=[1, ncols].
    columns = torch.arange(ncols, device=device).unsqueeze(0)  # 形状 (1, ncols)
    # build the mask for valid index by comparing columns and rt_row_lengths
    mask = columns < rt_row_lengths.unsqueeze(1)  # 形状 (nrows, ncols)
    # Get the corresponding indices for the inner value shape=[nrows, ncols]
    indices = starts + columns
    valid_rows, valid_cols = torch.where(mask)
    # Fill the output tensor with the mask and indices
    output[valid_rows, valid_cols] = inner_values[indices[mask]]
    return output


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


def gen_ragged_tensor(
    bs: int,
    min_seq: int,
    max_seq: int,
    min_n: int,
    max_n: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.int64,
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


class TestRaggedToDense(unittest.TestCase):
    """TestCase for verifying CPU/GPU consistency of ragged_to_dense"""

    def setUp(self):
        self.bs = 3500
        self.min_seq = 200
        self.max_seq = 300
        self.min_n = 20
        self.max_n = 30
        self.num_tests = 30

    def test_ragged_to_dense(self):
        for t in range(self.num_tests):
            seed = random.randrange(2**32)
            if t % 6 == 0:
                cpu_vals, cpu_offs, max_lengths = gen_ragged_tensor(
                    0,
                    self.min_seq,
                    self.max_seq,
                    self.min_n,
                    self.max_n,
                    device="cpu",
                    seed=seed,
                )
            elif t % 6 == 1:
                cpu_vals, cpu_offs, max_lengths = gen_ragged_tensor(
                    self.bs, self.min_seq, self.max_seq, 0, 0, device="cpu", seed=seed
                )
            elif t % 6 == 2:
                cpu_vals, cpu_offs, max_lengths = gen_ragged_tensor(
                    self.bs, 0, 0, 0, 0, device="cpu", seed=seed
                )
            elif t % 6 == 3:
                cpu_vals, cpu_offs, max_lengths = gen_2d_ragged_tensor(
                    self.bs, self.min_seq, self.max_seq, device="cpu", seed=seed
                )
                cpu_offs = [cpu_offs]
            elif t % 6 == 4:
                cpu_vals, cpu_offs, max_lengths = gen_2d_ragged_tensor(
                    self.bs, 0, 0, device="cpu", seed=seed
                )
                cpu_offs = [cpu_offs]
            else:
                cpu_vals, cpu_offs, max_lengths = gen_ragged_tensor(
                    self.bs,
                    self.min_seq,
                    self.max_seq,
                    self.min_n,
                    self.max_n,
                    device="cpu",
                    seed=seed,
                )
            cpu_out = ragged_to_dense(cpu_vals, cpu_offs, 0)
            gpu_vals = cpu_vals.cuda()
            gpu_offs = [o.cuda() for o in cpu_offs]
            gpu_out = ragged_to_dense(gpu_vals, gpu_offs, 0)

            torch.cuda.synchronize()
            torch_out = ragged_to_dense_torch(gpu_vals, gpu_offs)
            torch.cuda.synchronize()

            with self.subTest(test=t, seed=seed):
                self.assertTrue(
                    torch.equal(cpu_out, torch_out.cpu()),
                    f"torch and cpu Mismatch at test {t} with seed {seed}",
                )
                self.assertTrue(
                    torch.equal(cpu_out, gpu_out.cpu()),
                    f"cpu and gpu Mismatch at test {t} with seed {seed}",
                )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestRaggedToDense("test_ragged_to_dense"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
