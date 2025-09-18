import unittest

import torch

from recis.features.fused_op_impl import (
    FusedBoundaryOP,
    FusedCutoffOP,
    FusedHashOP,
    FusedModOP,
    FusedMultiHashOP,
)
from recis.features.op import Bucketize, Hash, IDMultiHash, Mod, SequenceTruncate
from recis.ragged.tensor import RaggedTensor


_FUSE_OP_NUM = 10


class TestFusedBoundaryOP(unittest.TestCase):
    def test_forward(self):
        boundaries = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
        inputs = []
        results = []
        fused_op = FusedBoundaryOP()
        for i in range(_FUSE_OP_NUM):
            boundaries_tmp = boundaries * (i + 1)
            fused_op.add_op(Bucketize(boundaries_tmp))
            values = torch.randn([10]) * 5 * (i + 1)
            data = RaggedTensor(values=values, offsets=torch.arange(0, 1001)).cuda()
            inputs.append(data)
            results.append(torch.bucketize(values.clone(), boundaries_tmp))
        outputs = fused_op.process(inputs)
        for output, ans in zip(outputs, results):
            self.assertTrue(torch.equal(output.values().cpu(), ans.cpu()))


class TestFusedUint64ModOP(unittest.TestCase):
    def test_forward(self):
        inputs = []
        results = []
        fused_op = FusedModOP()
        for i in range(_FUSE_OP_NUM):
            fused_op.add_op(Mod(1000))
            data = RaggedTensor(
                values=torch.tensor([1001, 1002, 1003]),
                offsets=torch.tensor([0, 1, 2, 3]),
            ).cuda()
            inputs.append(data)
            results.append(torch.tensor([1, 2, 3]))
        outputs = fused_op.process(inputs)
        for output, ans in zip(outputs, results):
            self.assertTrue(torch.equal(output.values().cpu(), ans.cpu()))


class TestFusedOP(unittest.TestCase):
    def test_hash_op(self):
        inputs = []
        results = []
        fused_op = FusedHashOP()
        ans_data = dict(
            farm=torch.LongTensor([-7736835464683084646, -5546720800891377920]),
            murmur=torch.LongTensor([-4003751876240412087, -1042754718167355375]),
        )

        for i in range(_FUSE_OP_NUM):
            for hash_type in ["farm", "murmur"]:
                fused_op.add_op(Hash(hash_type))
                values = torch.tensor([0, 1, 2, 1, 2, 3], dtype=torch.int8)
                data = RaggedTensor(
                    values=values,
                    offsets=[torch.tensor([0, 1]), torch.tensor([0, 3, 6])],
                ).cuda()
                inputs.append(data)
                ans = ans_data[hash_type]
                results.append(ans)
        outputs = fused_op.process(inputs)
        for output, ans in zip(outputs, results):
            self.assertTrue(torch.equal(output.values().cpu(), ans))


class TestFusedCutoffOP(unittest.TestCase):
    def test_forward_2d(self):
        inputs = []
        results = []
        fused_op = FusedCutoffOP()

        for i in range(_FUSE_OP_NUM):
            seq_len = 5 + i
            truncate_side = "left"
            dtype = torch.long if i % 2 == 0 else torch.float
            process_op = SequenceTruncate(
                seq_len=seq_len,
                check_length=False,
                truncate=True,
                truncate_side=truncate_side,
                n_dims=2,
                dtype=dtype,
            )
            fused_op.add_op(process_op)

            values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype)
            offsets = torch.tensor([0, 4, 7, 10]).int()
            data = RaggedTensor(values=values, offsets=offsets).cuda()
            inputs.append(data)

            single_output = process_op(data.clone())
            results.append(single_output)
        outputs = fused_op.process(inputs)

        for output, expected in zip(outputs, results):
            self.assertTrue(torch.equal(output.values().cpu(), expected.values().cpu()))
            for out_offset, exp_offset in zip(output.offsets(), expected.offsets()):
                self.assertTrue(torch.equal(out_offset.cpu(), exp_offset.cpu()))
            self.assertEqual(output._dense_shape, expected._dense_shape)

    def test_forward_3d(self):
        inputs = []
        results = []
        fused_op = FusedCutoffOP()

        for i in range(3):
            seq_len = 3 + i
            truncate_side = "left"
            dtype = torch.long if i % 2 == 0 else torch.float

            process_op = SequenceTruncate(
                seq_len=seq_len,
                check_length=False,
                truncate=True,
                truncate_side=truncate_side,
                n_dims=3,
                dtype=dtype,
            )
            fused_op.add_op(process_op)

            values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=dtype)
            offsets = [torch.tensor([0, 2, 3]).int(), torch.tensor([0, 3, 5, 8]).int()]
            data = RaggedTensor(values=values, offsets=offsets).cuda()
            inputs.append(data)

            single_output = process_op(data.clone())
            results.append(single_output)

        outputs = fused_op.process(inputs)

        for output, expected in zip(outputs, results):
            self.assertTrue(torch.equal(output.values().cpu(), expected.values().cpu()))
            for out_offset, exp_offset in zip(output.offsets(), expected.offsets()):
                self.assertTrue(torch.equal(out_offset.cpu(), exp_offset.cpu()))


class TestFusedYxMultiHashOP(unittest.TestCase):
    def test_forward(self):
        inputs = []
        results = []
        fused_op = FusedMultiHashOP()
        for i in range(_FUSE_OP_NUM):
            x = IDMultiHash([1000] * 3)
            fused_op.add_op(x)
            values = torch.tensor([1001, 1002, 1003], dtype=torch.int64)
            data = RaggedTensor(
                values=values,
                offsets=torch.tensor([0, 1, 2, 3]),
            ).cuda()
            inputs.append(data)
            results.append(x(data.clone()))
        outputs = fused_op.process(inputs)
        for output, ans in zip(outputs, results):
            for key in output.keys():
                self.assertTrue(
                    torch.equal(output[key].values().cpu(), ans[key].values().cpu())
                )


if __name__ == "__main__":
    unittest.main()
