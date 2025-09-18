import unittest

import torch

from recis.nn.modules.hashtable import HashTable


class HashTableGradAccumulateTest(unittest.TestCase):
    def test_grad(self):
        ht = HashTable(embedding_shape=[4], dtype=torch.float32)
        grad_index = torch.LongTensor([1, 2, 3])
        grad = torch.ones([3, 4], dtype=torch.float32)
        ht.accept_grad(grad_index, grad)
        grad_index = torch.LongTensor([1, 1, 1])
        grad = torch.ones([3, 4], dtype=torch.float32) * 2
        ht.accept_grad(grad_index, grad)
        grad_index = torch.LongTensor([2, 2, 2])
        grad = torch.ones([3, 4], dtype=torch.float32) * 3
        ht.accept_grad(grad_index, grad)
        ga_mean = ht.grad(3)
        indices = ga_mean.coalesce().indices()
        indices_true = torch.LongTensor([0, 1, 2])
        self.assertTrue(torch.allclose(indices, indices_true))
        values = ga_mean.coalesce().values()
        values_true = torch.tensor(
            [
                [7, 7, 7, 7],
                [10, 10, 10, 10],
                [1, 1, 1, 1],
            ]
        )
        values_true = values_true / 3.0
        self.assertTrue(torch.allclose(values, values_true))


if __name__ == "__main__":
    unittest.main()
