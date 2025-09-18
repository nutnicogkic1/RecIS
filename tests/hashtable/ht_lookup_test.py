import unittest

import torch

from recis.nn.hashtable_hook import AdmitHook
from recis.nn.modules.hashtable import HashTable


class CPUHashtableTest(unittest.TestCase):
    def setUp(self):
        self.ids_num = 2048
        self.emb_dim = 128
        self.block_size = 1024
        self.dtype = torch.float32

    def test_embedding_lookup(self):
        ht = HashTable(
            embedding_shape=[self.emb_dim],
            block_size=self.block_size,
            dtype=self.dtype,
            name="cpu_ht",
        )

        # init hashtable
        ids = torch.arange(self.ids_num, device="cuda")
        emb_r = ht(ids)
        exp_r = torch.zeros_like(emb_r)
        self.assertTrue((exp_r == emb_r).all())

        # insert datas
        ids_beg = self.ids_num
        ids = torch.arange(ids_beg, ids_beg + self.ids_num, device="cuda")
        emb = torch.tile(ids.reshape([-1, 1]), [1, self.emb_dim])
        ht._hashtable_impl.insert(ids, emb.type(self.dtype))
        emb_r = ht(ids)
        exp_r = emb
        self.assertTrue((exp_r == emb_r).all())

        # missing key
        ids_beg += self.ids_num
        ids = torch.arange(
            ids_beg - self.ids_num, ids_beg + self.ids_num, device="cuda"
        )
        emb_r = ht(ids)
        exp_r = torch.concat([emb, torch.zeros_like(emb)], 0)
        self.assertTrue((exp_r == emb_r).all())

        ids_r, _ = torch.sort(ht.ids())
        ids_exp = torch.arange(ids_beg + self.ids_num)
        self.assertTrue((ids_r == ids_exp).all())

    def test_embedding_lookup_readonly(self):
        ht = HashTable(
            embedding_shape=[self.emb_dim],
            block_size=self.block_size,
            dtype=self.dtype,
            name="cpu_ht_ro",
        )
        ro_hook = AdmitHook("ReadOnly")

        # init hashtable
        ids = torch.arange(self.ids_num, device="cuda")
        ht(ids, admit_hook=ro_hook)

        # insert datas
        ids_beg = self.ids_num
        ids = torch.arange(ids_beg, ids_beg + self.ids_num, device="cuda")
        emb = torch.tile(ids.reshape([-1, 1]), [1, self.emb_dim])
        ht._hashtable_impl.insert(ids, emb.type(self.dtype))
        ht(ids, admit_hook=ro_hook)

        # missing key
        ids_beg += self.ids_num
        ids = torch.arange(
            ids_beg - self.ids_num, ids_beg + self.ids_num, device="cuda"
        )
        ht(ids, admit_hook=ro_hook)

        ids_r, _ = torch.sort(ht.ids())
        ids_exp = torch.arange(self.ids_num, 2 * self.ids_num)
        self.assertTrue((ids_r == ids_exp).all())

    def test_embedding_lookup_read_only(self):
        ht = HashTable(
            embedding_shape=[self.emb_dim],
            block_size=self.block_size,
            dtype=self.dtype,
            name="cpu_ht_eval",
        )
        ht.eval()

        # init hashtable
        ids = torch.arange(self.ids_num, device="cuda")
        ht(ids)

        # insert datas
        ids_beg = self.ids_num
        ids = torch.arange(ids_beg, ids_beg + self.ids_num, device="cuda")
        emb = torch.tile(ids.reshape([-1, 1]), [1, self.emb_dim])
        ht._hashtable_impl.insert(ids, emb.type(self.dtype))
        ht(ids)

        # missing key
        ids_beg += self.ids_num
        ids = torch.arange(
            ids_beg - self.ids_num, ids_beg + self.ids_num, device="cuda"
        )
        ht(ids)

        ids_r, _ = torch.sort(ht.ids())
        ids_exp = torch.arange(self.ids_num, 2 * self.ids_num)
        self.assertTrue((ids_r == ids_exp).all())


class GPUHashtableTest(unittest.TestCase):
    def setUp(self):
        self.ids_num = 2048
        self.emb_dim = 128
        self.block_size = 1024
        self.dtype = torch.float32

    def test_embedding_lookup(self):
        ht = HashTable(
            embedding_shape=[self.emb_dim],
            block_size=self.block_size,
            dtype=self.dtype,
            device=torch.device("cuda"),
            name="gpu_ht",
        )

        # init hashtable
        ids = torch.arange(self.ids_num, device="cuda")
        emb_r = ht(ids)
        exp_r = torch.zeros_like(emb_r)
        self.assertTrue((exp_r == emb_r).all())

        # insert datas
        ids_beg = self.ids_num
        ids = torch.arange(ids_beg, ids_beg + self.ids_num, device="cuda")
        emb = torch.tile(ids.reshape([-1, 1]), [1, self.emb_dim])
        ht._hashtable_impl.insert(ids, emb.type(self.dtype))
        emb_r = ht(ids)
        exp_r = emb
        self.assertTrue((exp_r == emb_r).all())

        # missing key
        ids_beg += self.ids_num
        ids = torch.arange(
            ids_beg - self.ids_num, ids_beg + self.ids_num, device="cuda"
        )
        emb_r = ht(ids)
        exp_r = torch.concat([emb, torch.zeros_like(emb)], 0)
        self.assertTrue((exp_r == emb_r).all())

        ids_r, _ = torch.sort(ht.ids())
        ids_exp = torch.arange(ids_beg + self.ids_num, device="cuda")
        self.assertTrue((ids_r == ids_exp).all())

    def test_embedding_lookup_readonly(self):
        ht = HashTable(
            embedding_shape=[self.emb_dim],
            block_size=self.block_size,
            dtype=self.dtype,
            device=torch.device("cuda"),
            name="gpu_ht_ro",
        )
        ro_hook = AdmitHook("ReadOnly")
        # init hashtable
        ids = torch.arange(self.ids_num, device="cuda")
        ht(ids, admit_hook=ro_hook)

        # insert datas
        ids_beg = self.ids_num
        ids = torch.arange(ids_beg, ids_beg + self.ids_num, device="cuda")
        emb = torch.tile(ids.reshape([-1, 1]), [1, self.emb_dim])
        ht._hashtable_impl.insert(ids, emb.type(self.dtype))
        ht(ids, admit_hook=ro_hook)

        # missing key
        ids_beg += self.ids_num
        ids = torch.arange(
            ids_beg - self.ids_num, ids_beg + self.ids_num, device="cuda"
        )
        ht(ids, admit_hook=ro_hook)

        ids_r, _ = torch.sort(ht.ids())
        ids_exp = torch.arange(self.ids_num, 2 * self.ids_num, device="cuda")
        self.assertTrue((ids_r == ids_exp).all())

    def test_embedding_lookup_eval(self):
        ht = HashTable(
            embedding_shape=[self.emb_dim],
            block_size=self.block_size,
            dtype=self.dtype,
            device=torch.device("cuda"),
            name="gpu_ht_eval",
        )
        ht.eval()

        # init hashtable
        ids = torch.arange(self.ids_num, device="cuda")
        ht(ids)

        # insert datas
        ids_beg = self.ids_num
        ids = torch.arange(ids_beg, ids_beg + self.ids_num, device="cuda")
        emb = torch.tile(ids.reshape([-1, 1]), [1, self.emb_dim])
        ht._hashtable_impl.insert(ids, emb.type(self.dtype))
        ht(ids)

        # missing key
        ids_beg += self.ids_num
        ids = torch.arange(
            ids_beg - self.ids_num, ids_beg + self.ids_num, device="cuda"
        )
        ht(ids)

        ids_r, _ = torch.sort(ht.ids())
        ids_exp = torch.arange(self.ids_num, 2 * self.ids_num, device="cuda")
        self.assertTrue((ids_r == ids_exp).all())


if __name__ == "__main__":
    unittest.main()
