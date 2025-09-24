import unittest

import torch

from recis.nn.functional.embedding_ops import (
    segment_sum_sparse,
    sparse_embedding_segment_reduce,
    weight_norm_sparse,
)


def get_src_tensor(src, indices, num_segments):
    int_shape = src.shape
    out_shape = (num_segments, int_shape[1])
    out_tensor = torch.zeros(out_shape, dtype=src.dtype)
    for idx in range(len(indices)):
        val = indices[idx].item()
        out_tensor[idx] = src[val]
    return out_tensor


def tensor_weight(sum_tensor, weight_sum):
    for idx in range(len(weight_sum)):
        val = weight_sum[idx].item()
        if val != 0:
            sum_tensor[idx] = sum_tensor[idx] / weight_sum[idx]
    return sum_tensor


def segment_weight_mean_cpu(data, weight, indices, segment_ids, num_segments):
    assert data.dim() == 2, "dim must be 2"
    src_tensor = get_src_tensor(data, indices, num_segments)
    out_tensor = torch.zeros(src_tensor.shape, dtype=src_tensor.dtype)
    weight_sum = torch.zeros(num_segments, dtype=weight.dtype)
    for idx in range(segment_ids.numel()):
        seg_id = segment_ids[idx].item()
        w_val = weight[idx]
        src_val = src_tensor[idx]
        out_tensor[seg_id] += src_val * w_val
        weight_sum[seg_id] += weight[idx] * 1
    out = tensor_weight(out_tensor, weight_sum)
    return out


def segment_weight_sum_cpu(data, weight, indices, segment_ids, num_segments):
    assert data.dim() == 2, "dim must be 2"
    src_tensor = get_src_tensor(data, indices, num_segments)
    out_tensor = torch.zeros(src_tensor.shape, dtype=src_tensor.dtype)
    for idx in range(segment_ids.numel()):
        seg_id = segment_ids[idx].item()
        out_tensor[seg_id] += src_tensor[idx] * weight[idx]
    return out_tensor


def get_rand_tensor(shape, dtype=torch.float32, min=0.0, max=10.0):
    random_tensor = (max - min) * torch.rand(shape, dtype=torch.float32) + min
    random_tensor = random_tensor.to(dtype)
    return random_tensor


class TestSegmentOps(unittest.TestCase):
    def test_segment_weight_mean_cpu(self):
        data = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        weight = torch.tensor([2.0, 3.0, 0.5, 1.0])
        indices = torch.tensor([2, 0, 1, 0])
        segment_ids = torch.tensor([0, 1, 1, 2])
        num_segments = 4
        right_out = torch.tensor(
            [[3.0, 3.0], [4.0 / 3.5, 4.0 / 3.5], [1.0, 1.0], [0.0, 0]]
        )
        out = segment_weight_mean_cpu(data, weight, indices, segment_ids, num_segments)
        self.assertTrue(torch.allclose(right_out, out, atol=1e-7))  # assertEqual

    def test_segment_weight_sum_gpu(self):
        data = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        weight = torch.tensor([2.0, 3.0, 0.5, 1.0])
        indices = torch.tensor([2, 0, 1, 0])
        segment_ids = torch.tensor([0, 1, 1, 2])
        num_segments = 4
        data_g = data.cuda()
        weight_g = weight.cuda()
        indices_g = indices.cuda()
        segment_ids_g = segment_ids.cuda()
        gpu_out = segment_sum_sparse(
            data_g, weight_g, indices_g, segment_ids_g, num_segments
        )
        cpu_out = segment_weight_sum_cpu(
            data, weight, indices, segment_ids, num_segments
        )
        self.assertTrue(
            torch.allclose(gpu_out.cpu(), cpu_out, atol=1e-7)
        )  # assertEqual

    def test_segment_weight_mean_gpu(self):
        data = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        weight = torch.tensor([2.0, 3.0, 0.5, 1.0])
        indices = torch.tensor([2, 0, 1, 0])
        segment_ids = torch.tensor([0, 1, 1, 2])
        num_segments = 4
        data_g = data.cuda()
        weight_g = weight.cuda()
        indices_g = indices.cuda()
        segment_ids_g = segment_ids.cuda()

        gpu_weight_norm = weight_norm_sparse(
            data_g, weight_g, segment_ids_g, num_segments
        )
        gpu_out = segment_sum_sparse(
            data_g, gpu_weight_norm, indices_g, segment_ids_g, num_segments
        )
        cpu_out = segment_weight_mean_cpu(
            data, weight, indices, segment_ids, num_segments
        )

        self.assertTrue(
            torch.allclose(gpu_out.cpu(), cpu_out, atol=1e-7)
        )  # assertEqual

    def test_segment_weight_sum_rand(self):
        dim = 128
        num_segments = 512
        data_rows = int(num_segments / 2)
        ids_rows = int(num_segments / 3 * 2)
        data = get_rand_tensor(
            shape=(data_rows, dim), dtype=torch.float32, min=0, max=1
        )
        weight = get_rand_tensor(shape=(ids_rows), dtype=torch.float32, min=0, max=1)
        indices = get_rand_tensor(
            shape=(ids_rows), dtype=torch.int64, min=0, max=data_rows
        )
        segment_ids = get_rand_tensor(
            shape=(ids_rows), dtype=torch.int64, min=0, max=num_segments
        )

        data_g = data.cuda()
        weight_g = weight.cuda()
        indices_g = indices.cuda()
        segment_ids_g = segment_ids.cuda()
        gpu_out = segment_sum_sparse(
            data_g, weight_g, indices_g, segment_ids_g, num_segments
        )
        cpu_out = segment_weight_sum_cpu(
            data, weight, indices, segment_ids, num_segments
        )
        self.assertTrue(torch.allclose(gpu_out.cpu(), cpu_out, atol=1e-7))

    def test_segment_weight_sum_empty(self):
        num_segments = 512
        ids_rows = int(num_segments / 3 * 2)
        data = torch.empty((0, 0), dtype=torch.float32)
        weight = get_rand_tensor(shape=(ids_rows), dtype=torch.float32, min=0, max=1)
        indices = torch.empty((0), dtype=torch.int64)
        segment_ids = get_rand_tensor(
            shape=(ids_rows), dtype=torch.int64, min=0, max=num_segments
        )

        data_g = data.cuda()
        weight_g = weight.cuda()
        indices_g = indices.cuda()
        segment_ids_g = segment_ids.cuda()
        gpu_out = segment_sum_sparse(
            data_g, weight_g, indices_g, segment_ids_g, num_segments
        )
        cpu_out = segment_weight_sum_cpu(
            data, weight, indices, segment_ids, num_segments
        )
        self.assertTrue(torch.allclose(gpu_out.cpu(), cpu_out, atol=1e-7))

    def test_segment_weight_mean_rand(self):
        dim = 16
        num_segments = 512
        data_rows = int(num_segments / 2)
        ids_rows = int(num_segments / 3 * 2)
        data = get_rand_tensor(
            shape=(data_rows, dim), dtype=torch.float32, min=0, max=1
        )
        weight = get_rand_tensor(
            shape=(ids_rows), dtype=torch.float32, min=0.1, max=0.5
        )
        indices = get_rand_tensor(
            shape=(ids_rows), dtype=torch.int64, min=0, max=data_rows
        )
        segment_ids = get_rand_tensor(
            shape=(ids_rows), dtype=torch.int64, min=0, max=num_segments
        )
        data_g = data.cuda()
        weight_g = weight.cuda()
        indices_g = indices.cuda()
        segment_ids_g = segment_ids.cuda()
        gpu_weight_norm = weight_norm_sparse(
            data_g, weight_g, segment_ids_g, num_segments
        )
        gpu_out = segment_sum_sparse(
            data_g, gpu_weight_norm, indices_g, segment_ids_g, num_segments
        )
        cpu_out = segment_weight_mean_cpu(
            data, weight, indices, segment_ids, num_segments
        )
        torch.set_printoptions(threshold=1000)
        self.assertTrue(torch.allclose(gpu_out.cpu(), cpu_out, atol=1e-7))

    def test_segment_weight_sum_grad(self):
        num_segments = 4
        data = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        weight = torch.tensor([2.0, 3.0, 0.5, 1.0])
        indices = torch.tensor([2, 0, 1, 0])
        segment_ids = torch.tensor([0, 1, 1, 2])
        grad = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        data_grad = torch.tensor([[4.0, 4.0], [0.5, 0.5], [2.0, 2.0]])

        data_g = data.cuda()
        weight_g = weight.cuda()
        indices_g = indices.cuda()
        segment_ids_g = segment_ids.cuda()
        grad_g = grad.cuda()

        data_g.requires_grad_()
        data_g.retain_grad()
        gpu_out = sparse_embedding_segment_reduce(
            data_g, weight_g, indices_g, segment_ids_g, num_segments, "sum"
        )
        gpu_out.backward(grad_g, retain_graph=True)
        gpu_grad = data_g.grad
        self.assertTrue(torch.allclose(gpu_grad.cpu(), data_grad, atol=1e-7))

    def test_segment_weight_mean_grad(self):
        num_segments = 4
        data = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        weight = torch.tensor([2.0, 3.0, 0.5, 1.0])
        indices = torch.tensor([2, 0, 1, 0])
        segment_ids = torch.tensor([0, 1, 1, 2])
        grad = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        data_grad = torch.tensor(
            [[1.0 + 3.0 / 3.5, 1.0 + 3.0 / 3.5], [0.5 / 3.5, 0.5 / 3.5], [1.0, 1.0]]
        )

        data_g = data.cuda()
        weight_g = weight.cuda()
        indices_g = indices.cuda()
        segment_ids_g = segment_ids.cuda()
        grad_g = grad.cuda()
        data_g.requires_grad_()
        data_g.retain_grad()
        gpu_out = sparse_embedding_segment_reduce(
            data_g, weight_g, indices_g, segment_ids_g, num_segments, "mean"
        )
        gpu_out.backward(grad_g, retain_graph=True)
        gpu_grad = data_g.grad
        self.assertTrue(torch.allclose(gpu_grad.cpu(), data_grad, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
