import unittest

import torch

import recis


def ids_partition(
    ids: torch.tensor, world_size: int = 1, max_partition_num: int = 65536
):
    assert max_partition_num == 65536
    ids, reverse_indice = torch.unique(ids, return_inverse=True)
    ids_module = torch.bitwise_and(ids, torch.tensor(65535, device=ids.device))
    arg_pos = torch.argsort(ids_module)
    ids_module_sorted = ids_module[arg_pos]
    m, n = divmod(max_partition_num, world_size)
    parts = [m + (i < n) for i in range(world_size)]
    parts_boundary = torch.cumsum(torch.tensor(parts, device=ids.device), dim=0)
    boundary_indices = torch.searchsorted(ids_module_sorted, parts_boundary)
    boundary_indices = torch.cat(
        (torch.tensor([0], device=ids.device), boundary_indices)
    )
    segment_size = boundary_indices[1:] - boundary_indices[:-1]
    range_index = torch.argsort(arg_pos)
    return ids[arg_pos], segment_size, range_index[reverse_indice]


class IDSPartitionTest(unittest.TestCase):
    def get_data(self, device="cpu", ids_count=1000):
        ids = torch.randint(
            0, ids_count * 100, (ids_count,), dtype=torch.int64, device=device
        )
        return ids

    def test_ids_encode_cpu(self):
        ids = self.get_data("cpu")
        world_size = 128
        max_partition_num = 65536
        recis_result = torch.ops.recis.ids_partition(ids, world_size)
        torch_result = ids_partition(ids, world_size, max_partition_num)
        for i in range(len(recis_result)):
            self.assertTrue(torch.equal(recis_result[i], torch_result[i]))

    def test_ids_encode_cuda(self):
        for count in [10, 100, 10000, 1000000]:
            with self.subTest(id_count=count):
                ids = self.get_data("cuda", count)
                world_size = 256
                torch.cuda.synchronize()
                recis_result = torch.ops.recis.ids_partition(ids, world_size)
                torch_result = ids_partition(ids, world_size, 65536)
                torch.cuda.synchronize()
                for i in range(len(recis_result)):
                    self.assertTrue(
                        torch.equal(recis_result[i].cpu(), torch_result[i].cpu()),
                        f"{count}, recis_result is {recis_result[i]}, torch_result is {torch_result[i]}",
                    )


if __name__ == "__main__":
    print(recis.__version__)
    unittest.main()
