import unittest

import torch

import recis


class TestGenerateIds(unittest.TestCase):
    def setUp(self):
        self.gen_num = 10000
        self.free_size = 2000
        self.cur_size = 101
        self.free_block = [torch.arange(100000, dtype=torch.int64, device="cuda")]
        print(recis.__version__)

    def test_generate_ids(self):
        ret = torch.ops.recis.generate_ids(
            self.gen_num, self.free_block, 0, self.cur_size, self.free_block[0].numel()
        )
        ans = torch.arange(101, 101 + self.gen_num, device=ret.device).flip([0])
        self.assertTrue(torch.equal(ret, ans))

    def test_generate_ids_with_free_ids(self):
        ret = torch.ops.recis.generate_ids(
            self.gen_num,
            self.free_block,
            self.free_size,
            self.cur_size,
            self.free_block[0].numel(),
        )
        ans = [
            torch.arange(
                101, 101 + self.gen_num - self.free_size, device=ret.device
            ).flip([0])
        ]
        ans.append(self.free_block[0][:2000])
        ans = torch.cat(ans)
        self.assertTrue(torch.equal(ret, ans))


class TestFreeIds(unittest.TestCase):
    def setUp(self):
        self.free_num = 10000
        self.free_size = 2000
        self.free_block = [torch.arange(100000, dtype=torch.int64, device="cuda")]

    def test_free_ids(self):
        free_ids = torch.arange(101, 101 + self.free_num, device="cuda")
        torch.ops.recis.free_ids(
            free_ids, self.free_block, self.free_size, self.free_block[0].numel()
        )
        ans = torch.arange(100000, dtype=torch.int64, device="cuda")
        ans[2000 : 2000 + 10000] = free_ids
        self.assertTrue(torch.equal(self.free_block[0], ans))


class TestMaskIndex(unittest.TestCase):
    def setUp(self):
        self.total_num = 10000
        self.mask_ratio = 0.2
        self.mask = torch.empty(
            (self.total_num,), dtype=torch.float32, device="cuda"
        ).uniform_() * (1 + self.mask_ratio)
        self.mask = torch.floor(self.mask).to(torch.bool)
        self.id = torch.arange(101, 101 + self.total_num, device="cuda")
        self.index = torch.arange(101, 101 + self.total_num, device="cuda").flip([0])

    def test_mask_index(self):
        gen_cumsum_num = torch.cumsum(self.mask, 0)
        gen_num = gen_cumsum_num[-1].item()
        new_ids, range_index = torch.ops.recis.mask_key_index(
            self.id, self.mask, gen_cumsum_num, gen_num
        )
        new_index = torch.arange(5000, 5000 + gen_num, device="cuda")
        index = torch.ops.recis.scatter_ids_with_mask(
            self.index, new_index, range_index
        )

        new_ids_ans = torch.masked_select(self.id, self.mask)
        range_index_ans = torch.arange(self.total_num, device="cuda")
        range_index_ans = torch.masked_select(range_index_ans, self.mask)
        index_ans = self.index.clone()
        index_ans[range_index_ans] = new_index
        self.assertTrue(torch.equal(new_ids, new_ids_ans))
        self.assertTrue(torch.equal(range_index, range_index_ans))
        self.assertTrue(torch.equal(index, index_ans))


if __name__ == "__main__":
    unittest.main()
