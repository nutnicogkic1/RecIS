import unittest

import torch

from recis.features.op import (
    Bucketize,
    FeatureCross,
    Hash,
    IDMultiHash,
    Mod,
    SequenceTruncate,
)
from recis.ragged.tensor import RaggedTensor


class TestUint64Mod(unittest.TestCase):
    def test_uint64_mod(self):
        op = Mod(1000)
        data = RaggedTensor(
            values=torch.tensor([1001, 1002, 1003]),
            offsets=torch.tensor([0, 1, 2, 3]).cuda(),
        )
        output = op(data)
        self.assertTrue(torch.equal(output.values().cpu(), torch.tensor([1, 2, 3])))


class TestBoundary(unittest.TestCase):
    def test_boundary(self):
        boundaries = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
        op = Bucketize(boundaries)
        values = torch.randn([10]) * 5
        data = RaggedTensor(values=values, offsets=torch.arange(0, 1001)).cuda()
        output = op(data)
        ans = torch.bucketize(values.clone(), torch.tensor([1, 2, 3, 4, 5]))
        self.assertTrue(torch.equal(output.values().cpu(), ans))


class TestFeatureCross(unittest.TestCase):
    def test_feature_cross(self):
        op = FeatureCross()
        x = RaggedTensor(
            values=torch.Tensor([1001, 1002, 1003]).long(),
            offsets=torch.Tensor([0, 1, 2, 3]).int(),
            weight=torch.ones(3),
        ).cuda()
        y = RaggedTensor(
            values=torch.tensor([1, 2, 3]).long(),
            offsets=torch.tensor([0, 1, 2, 3]).int(),
            weight=torch.ones(3),
        ).cuda()
        output = op([x, y])
        self.assertTrue(
            torch.equal(
                output.values().cpu(),
                torch.LongTensor(
                    [-7808879811642834040, -4967483181219983279, 1752425377096056168]
                ),
            )
        )
        self.assertTrue(
            torch.equal(output.offsets()[-1].cpu(), torch.LongTensor([0, 1, 2, 3]))
        )


class TestProcessSequences(unittest.TestCase):
    def test_process_sequences_check_length(self):
        op = SequenceTruncate(seq_len=128, check_length=True)
        data = RaggedTensor(
            values=torch.randn([2 * 256]),
            offsets=[
                torch.IntTensor([0, 256, 512]),
                torch.arange(0, 513, dtype=torch.int32),
            ],
        ).cuda()
        try:
            op(data)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

    def test_process_sequences_truncate(self):
        op = SequenceTruncate(
            seq_len=128, check_length=False, truncate=True, truncate_side="left"
        )
        data = RaggedTensor(
            values=torch.randn([2 * 256]), offsets=[torch.IntTensor([0, 256, 512])]
        ).cuda()
        output = op(data)
        self.assertTrue(
            torch.equal(output.offsets()[-1].cpu(), torch.IntTensor([0, 128, 256]))
        )


class TestMultiHash(unittest.TestCase):
    def test_y_x_multi_hash(self):
        op = IDMultiHash([1000] * 4)
        ids = RaggedTensor(
            values=torch.tensor([1001, 1002, 1003]), offsets=torch.tensor([0, 1, 2, 3])
        ).cuda()
        ret = op(ids)
        ans = [
            torch.LongTensor([1, 2, 3]),
            torch.LongTensor([3, 6, 9]),
            torch.LongTensor([5, 10, 15]),
            torch.LongTensor([7, 14, 21]),
        ]
        for i in range(4):
            data = ret[f"multi_hash_{i}"]
            self.assertTrue(torch.equal(data.values().cpu(), ans[i]))


class TestHashOP(unittest.TestCase):
    def test_hash_op(self):
        ans = [
            torch.LongTensor([-7736835464683084646, -5546720800891377920]),
            torch.LongTensor([-4003751876240412087, -1042754718167355375]),
        ]
        i = 0
        for hash_type in ["farm", "murmur"]:
            with self.subTest(hash_type=hash_type):
                op = Hash(hash_type)
                values = torch.tensor([0, 1, 2, 1, 2, 3], dtype=torch.int8)
                data = RaggedTensor(
                    values=values,
                    offsets=[torch.tensor([0, 1]), torch.tensor([0, 3, 6])],
                ).cuda()
                output = op(data)
                self.assertTrue(torch.equal(output.values().cpu(), ans[i]))
                i += 1


if __name__ == "__main__":
    unittest.main()
