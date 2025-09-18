import unittest
import uuid

import torch

from recis.nn.modules.hashtable import HashTable


class CPUHashtableSlotTest(unittest.TestCase):
    def setUp(self):
        self.ids_num = 2048
        self.emb_dim = 128
        self.block_size = 1024
        self.dtype = torch.float32

    def test_embedding_lookup(self):
        ht = HashTable(
            embedding_shape=[self.emb_dim],
            children=[uuid.uuid4().hex],
            block_size=self.block_size,
            dtype=self.dtype,
            name="cpu_ht",
        )

        # insert datas
        ids = torch.arange(self.ids_num, device="cuda")
        emb = torch.tile(ids.reshape([-1, 1]), [1, self.emb_dim])
        ht._hashtable_impl.insert(ids, emb.type(self.dtype))
        ht(ids)
        slot_group = ht.slot_group()
        slot_emb = slot_group.slot_by_name("embedding")

        self.assertTrue(slot_emb.block_size() == self.block_size)
        self.assertTrue(slot_emb.name() == "embedding")
        self.assertTrue(slot_emb.flat_size() == self.emb_dim)

        exp_slot = torch.cat(
            [
                emb,
                torch.zeros(
                    (self.block_size, self.emb_dim), dtype=emb.dtype, device=emb.device
                ),
            ],
            dim=0,
        ).cpu()
        self.assertTrue(torch.equal(slot_emb.value().sum(), exp_slot.sum()))


class GPUHashtableSlotTest(unittest.TestCase):
    def setUp(self):
        self.ids_num = 2048
        self.emb_dim = 128
        self.block_size = 1024
        self.dtype = torch.float32

    def test_embedding_lookup(self):
        ht = HashTable(
            embedding_shape=[self.emb_dim],
            block_size=self.block_size,
            children=[uuid.uuid4().hex],
            device=torch.device("cuda"),
            dtype=self.dtype,
            name="gpu_ht",
        )

        # insert datas
        ids = torch.arange(self.ids_num, device="cuda")
        emb = torch.tile(ids.reshape([-1, 1]), [1, self.emb_dim])
        ht._hashtable_impl.insert(ids, emb.type(self.dtype))
        ht(ids)
        slot_group = ht.slot_group()
        slot_emb = slot_group.slot_by_name("embedding")

        self.assertTrue(slot_emb.block_size() == self.block_size)
        self.assertTrue(slot_emb.name() == "embedding")
        self.assertTrue(slot_emb.flat_size() == self.emb_dim)

        exp_slot = torch.cat(
            [
                emb,
                torch.zeros(
                    (self.block_size, self.emb_dim), dtype=emb.dtype, device=emb.device
                ),
            ],
            dim=0,
        )
        self.assertTrue(torch.equal(slot_emb.value().sum(), exp_slot.sum()))


if __name__ == "__main__":
    unittest.main()
