import unittest

import torch


def get_invaild_data(device="cpu", dtype=torch.float32):
    data = (
        torch.Tensor([[1, 0, 3, 0, 0], [6, 7, 8, 0, 0]])
        .to(device=device)
        .to(dtype=dtype)
    )
    values = torch.Tensor([1, 0, 3, 6, 7, 8]).to(device=device).to(dtype=dtype)
    offsets = torch.tensor([0, 3, 6], device=device).to(torch.int)
    return data, values, offsets


def get_data(device="cpu", dtype=torch.float32):
    data = (
        torch.Tensor([[1, 2, 3, 0, 0], [6, 7, 8, 0, 0]])
        .to(device=device)
        .to(dtype=dtype)
    )
    values = torch.Tensor([1, 2, 3, 6, 7, 8]).to(device=device).to(dtype=dtype)
    offsets = torch.tensor([0, 3, 6], device=device).to(torch.int)
    return data, values, offsets


def dense_to_ragged(data, check_invalid, invalid_value):
    invalid_tensor = torch.Tensor([invalid_value]).to(dtype=data.dtype)
    return torch.ops.recis.dense_to_ragged(data, check_invalid, invalid_tensor)


class TestRaggedToDense(unittest.TestCase):
    def test_ragged_to_dense_check_invalid(self):
        check_invalid = True
        invalid_value = 0
        for data_invalid in [True, False]:
            for device in ["cpu", "cuda"]:
                for dtype in [torch.int32, torch.float32, torch.int64]:
                    with self.subTest(
                        device=device,
                        dtype=dtype,
                        check_invalid=check_invalid,
                        data_invalid=data_invalid,
                    ):
                        if data_invalid:
                            data, values, offsets = get_invaild_data(device, dtype)
                        else:
                            data, values, offsets = get_data(device, dtype)
                        values_ret, offsets_ret = dense_to_ragged(
                            data, check_invalid, invalid_value
                        )
                        self.assertTrue(torch.equal(values, values_ret))
                        self.assertTrue(torch.equal(offsets, offsets_ret))

    def test_ragged_to_dense_no_check_invalid(self):
        check_invalid = False
        invalid_value = 0
        for device in ["cpu"]:
            for dtype in [torch.int32]:
                with self.subTest(
                    device=device, dtype=dtype, check_invalid=check_invalid
                ):
                    print(device, dtype, "test_ragged_to_dense")
                    data, _, _ = get_data(device, dtype)
                    values_ret, offsets_ret = dense_to_ragged(
                        data, check_invalid, invalid_value
                    )
                    self.assertTrue(torch.equal(data.view(-1), values_ret))
                    rows = data.shape[0]
                    cols = data.shape[1]
                    offsets = torch.arange(0, rows + 1, cols, device=device).to(
                        torch.int
                    )
                    self.assertTrue(torch.equal(offsets, offsets_ret))


if __name__ == "__main__":
    unittest.main()
