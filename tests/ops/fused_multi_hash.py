import math
import random
import unittest

import torch

from recis.nn.functional.fused_ops import fused_multi_hash


def _is_prime(x):
    for n in range(int(math.sqrt(x) + 1e-6), 1, -1):
        if x % n == 0:
            return False
    return True


def _find_prime_lower_than(x):
    for n in range(x, 0, -1):
        if _is_prime(n):
            return n
    return 11


def gen_data(device="cpu"):
    multi_muls = [1, 3, 5, 7]
    multi_mods = [29, 47, 67, 83]
    tensor_nums = 10
    tensors = []
    ans = []
    multi_muls_tensors = []
    num_buckets_tensors = []
    multi_primes_tensors = []
    for i in range(tensor_nums):
        x = torch.randint(0, 100, (100000,), dtype=torch.int64, device=device)
        num_buckets = [random.randint(10000, 1000000) for _ in range(4)]
        multi_primes = [
            _find_prime_lower_than(multi_mods[i] * num_buckets[i]) for i in range(4)
        ]
        tensors.append(x)
        for mul, primes, bucket in zip(multi_muls, multi_primes, num_buckets):
            out = (((x * mul) % primes + primes) % primes) % bucket
            ans.append(out)
        multi_muls_tensors.append(
            torch.tensor(multi_muls, device=device, dtype=torch.int64)
        )
        num_buckets_tensors.append(
            torch.tensor(num_buckets, device=device, dtype=torch.int64)
        )
        multi_primes_tensors.append(
            torch.tensor(multi_primes, device=device, dtype=torch.int64)
        )
    return tensors, ans, multi_muls_tensors, num_buckets_tensors, multi_primes_tensors


class TestFusedHash(unittest.TestCase):
    def test_fused_multi_hash_cpu(self):
        tensors, ans, multi_muls_tensors, num_buckets_tensors, multi_primes_tensors = (
            gen_data()
        )
        outputs = fused_multi_hash(
            tensors, multi_muls_tensors, multi_primes_tensors, num_buckets_tensors
        )
        for out, a in zip(outputs, ans):
            self.assertTrue(torch.allclose(out, a))

    def test_fused_multi_hash_gpu(self):
        tensors, ans, multi_muls_tensors, num_buckets_tensors, multi_primes_tensors = (
            gen_data(device="cuda")
        )
        outputs = fused_multi_hash(
            tensors, multi_muls_tensors, multi_primes_tensors, num_buckets_tensors
        )
        for out, a in zip(outputs, ans):
            self.assertTrue(torch.allclose(out, a))


if __name__ == "__main__":
    unittest.main()
