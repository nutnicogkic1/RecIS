import unittest

import torch

from recis.nn.functional import hash_ops


class FusedHashTest(unittest.TestCase):
    def test_fused_farmhash(self):
        inputs = [
            torch.tensor([0, 1, 2, 1, 2, 3], dtype=torch.int8).cuda(),
            torch.tensor([0, 5, 5, 5, 6, 7], dtype=torch.int8).cuda(),
        ]
        offsets = [
            torch.tensor([0, 3, 6], dtype=torch.int32).cuda(),
            torch.tensor([0, 3, 6], dtype=torch.int32).cuda(),
        ]
        outputs = hash_ops.farmhash(inputs, offsets)
        res = [
            torch.LongTensor([-7736835464683084646, -5546720800891377920]),
            torch.LongTensor([-4442022684725028445, -909183002746659612]),
        ]
        for output, ans in zip(outputs, res):
            self.assertTrue(torch.equal(output.cpu(), ans))

    def test_fused_murmurhash(self):
        inputs = [
            torch.tensor([0, 1, 2, 1, 2, 3], dtype=torch.int8).cuda(),
            torch.tensor([0, 5, 5, 5, 6, 7], dtype=torch.int8).cuda(),
        ]
        offsets = [
            torch.tensor([0, 3, 6], dtype=torch.int32).cuda(),
            torch.tensor([0, 3, 6], dtype=torch.int32).cuda(),
        ]
        outputs = hash_ops.murmurhash(inputs, offsets)
        res = [
            torch.LongTensor([-4003751876240412087, -1042754718167355375]),
            torch.LongTensor([-5541886947548579749, 4434088491466393040]),
        ]
        for output, ans in zip(outputs, res):
            self.assertTrue(torch.equal(output.cpu(), ans))


if __name__ == "__main__":
    unittest.main()
