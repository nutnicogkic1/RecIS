import unittest
from typing import List

import numpy as np
import torch

import recis.nn.functional.ragged_ops as ragged


def get_table(rows, cols, random=False):
    matrix = []
    if not random:
        for i in range(rows):
            row = []
            for j in range(cols):
                value = i + j / 100.0
                row.append(value)
            matrix.append(row)
        matrix = np.array(matrix, dtype=np.float32)
    else:
        matrix = np.random.rand(rows, cols).astype(np.float32)
    return matrix


def get_dy(batch, seq, dim, random=False):
    row = np.array(batch) * np.array(seq)
    cum = np.cumsum(row).tolist()
    rows = cum[-1]
    dy = get_table(rows, dim, random)
    return dy


def splite_dy(batch, seq, dy):
    dys = []
    row = np.array(batch) * np.array(seq)
    cum = [0] + np.cumsum(row).tolist()
    for idx in range(len(cum) - 1):
        start = cum[idx]
        end = cum[idx + 1]
        cur_dy = dy[start:end]
        dys.append(cur_dy)
    return dys


def combin_offset(offset):
    out_off = offset[0].copy()
    for idx in range(1, len(offset)):
        compen = out_off[-1]
        cur = offset[idx]
        out = cur[1:]
        out = [x + compen for x in out]
        out_off += out
    return out_off


def splite_off(batch, offset):
    bt = batch.copy()
    bt[0] += 1
    batch_cum = [0] + np.cumsum(bt).tolist()
    out = []
    last_off = 0
    for idx in range(1, len(batch_cum)):
        cur_off = offset[batch_cum[idx - 1] : batch_cum[idx]]
        if idx >= 2:
            cur_off = [last_off] + cur_off
        last_off = cur_off[-1]
        out.append(cur_off)
    return out


def tile_row_cpu(offset, seq, table):
    table_row, dim = table.shape
    out = []
    arr_zero = np.zeros(dim, dtype=table.dtype)  # used to null value to padding
    for idx in range(0, len(offset) - 1):
        row_start = offset[idx]
        row_end = offset[idx + 1]
        row_len = row_end - row_start
        for id in range(seq):
            cur = None
            if id < row_len:
                cur = table[row_start + id, :]
            else:
                cur = arr_zero
            out.append(cur)
    return out


def tile_back_row_cpu(offset, seq, indices, dy, d_table):
    for idx in range(len(offset) - 1):
        start = offset[idx]
        end = offset[idx + 1]
        len_off = end - start
        lens = min(len_off, seq)
        for id in range(lens):
            indice = indices[start + id]
            cur_dy = dy[idx][id]
            d_table[indice, :] += cur_dy


def restore_table(table, value):
    out = []
    row, col = table.shape
    for idx in range(len(value)):
        val = value[idx]
        assert val < row
        obj = table[val, :]
        out.append(obj)
    return np.array(out)


def tile_cpu(
    value: List[int], offset: List[int], seq: List[int], table: np.ndarray
) -> np.ndarray:
    out = []
    new_table = restore_table(table, value)
    for idx in range(len(offset)):
        rt = tile_row_cpu(offset[idx], seq[idx], new_table)
        out = out + rt
    return np.array(out)


def tile_backward_cpu(
    value: List[int],
    offset: List[int],
    seq: List[int],
    dy: List[np.ndarray],
    dx_shape: tuple,
) -> np.ndarray:
    dx = np.zeros(dx_shape, dy[0].dtype)
    for idx in range(len(offset)):
        cur_off = offset[idx]
        shape = (len(cur_off) - 1, seq[idx], dx_shape[1])
        cur_dy = dy[idx].reshape(shape)
        tile_back_row_cpu(cur_off, seq[idx], value, cur_dy, dx)
    return dx


def tile_para():
    batch = [3, 3, 4]
    seq = [3, 4, 5]
    offset1 = [0, 1, 3, 4]
    offset2 = [0, 1, 4, 5]
    offset3 = [0, 1, 5, 6, 7]
    offset = [offset1, offset2, offset3]
    dim = 2
    combin_off = combin_offset(offset)
    value = list(range(max(combin_off)))
    val_max = max(value)
    table = get_table(val_max + 1, dim)
    dy = get_dy(batch, seq, dim)
    out = {
        "batch": batch,
        "seq": seq,
        "offset": combin_off,
        "value": value,
        "table": table,
        "dy": dy,
    }
    return out


def tile_para_random():
    batch_min = 100
    batch_max = 120
    tensor_num = 30
    seq_min = 16
    seq_max = 1024
    value_len = 1024 * 4
    value_max = 512
    dim = 32
    batch = np.random.randint(
        batch_min, batch_max, size=tensor_num, dtype=np.int32
    ).tolist()
    seq = np.random.randint(seq_min, seq_max, size=tensor_num, dtype=np.int32).tolist()
    batch_lens = np.sum(batch).item()
    offset = np.random.randint(0, value_len, size=batch_lens + 1, dtype=np.int32)
    offset = np.sort(offset).tolist()
    offset[0] = 0
    value = np.random.randint(0, value_max, size=max(offset), dtype=np.int32)
    value = np.sort(value).tolist()
    value_max = max(value)
    table = get_table(value_max + 1, dim, True)
    dy = get_dy(batch, seq, dim, True)
    out = {
        "batch": batch,
        "seq": seq,
        "offset": offset,
        "value": value,
        "table": table,
        "dy": dy,
    }
    return out


def tile_input(random=False):
    if random:
        return tile_para_random()
    else:
        return tile_para()


def para_impro(para):
    para_cpu = para.copy()
    para_gpu = para.copy()
    para_cpu["offset"] = splite_off(para_cpu["batch"], para_cpu["offset"])
    para_cpu["dy"] = splite_dy(para_cpu["batch"], para_cpu["seq"], para_cpu["dy"])
    device = "cuda"
    para_gpu["offset"] = torch.tensor(para_gpu["offset"], device=device)
    para_gpu["value"] = torch.tensor(para_gpu["value"], device=device)
    para_gpu["table"] = torch.tensor(para_gpu["table"], device=device)
    para_gpu["dy"] = torch.tensor(para_gpu["dy"], device=device)
    return (para_cpu, para_gpu)


class TestRaggedTileOp(unittest.TestCase):
    def setUp(self):
        para = tile_input(random=True)
        self.para = para_impro(para)

    def test_forward(self):
        tile_func = tile_cpu
        para_cpu, para_gpu = self.para
        x_cpu = tile_func(
            para_cpu["value"], para_cpu["offset"], para_cpu["seq"], para_cpu["table"]
        )
        x_gpu = ragged.ragged_tile(
            para_gpu["batch"],
            para_gpu["seq"],
            para_gpu["value"],
            para_gpu["offset"],
            para_gpu["table"],
        )
        torch.allclose(torch.tensor(x_cpu), x_gpu.cpu(), atol=1e-5)

    def test_backward(self):
        para_cpu, para_gpu = self.para
        para_gpu["table"].requires_grad_(True)
        dx_cpu = tile_backward_cpu(
            para_cpu["value"],
            para_cpu["offset"],
            para_cpu["seq"],
            para_cpu["dy"],
            para_cpu["table"].shape,
        )
        y_gpu = ragged.ragged_tile(
            para_gpu["batch"],
            para_gpu["seq"],
            para_gpu["value"],
            para_gpu["offset"],
            para_gpu["table"],
        )
        y_gpu.backward(para_gpu["dy"])
        dx_gpu = para_gpu["table"].grad
        torch.allclose(torch.tensor(dx_cpu), dx_gpu.cpu(), atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
