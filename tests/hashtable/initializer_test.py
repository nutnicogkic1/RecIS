import unittest

import torch

from recis.nn.initializers import (
    ConstantInitializer,
    KaimingNormalInitializer,
    KaimingUniformInitializer,
    NormalInitializer,
    TruncNormalInitializer,
    UniformInitializer,
    XavierNormalInitializer,
    XavierUniformInitializer,
)


class TestInitializers(unittest.TestCase):
    @staticmethod
    def set_and_build_initializer(initializer: ConstantInitializer):
        initializer.set_shape([8, 8])
        initializer.set_dtype()
        initializer.build()

    def test_constant(self):
        initializer = ConstantInitializer()
        self.set_and_build_initializer(initializer)
        ret = initializer.generate()
        self.assertTrue(torch.allclose(ret, torch.zeros_like(ret), atol=1e-7))

    def test_uniform(self):
        generator = torch.Generator(torch.device("cpu"))
        generator.manual_seed(0)
        initializer = UniformInitializer(generator=generator)
        self.set_and_build_initializer(initializer)
        ret_one = initializer.generate()
        ret_two = initializer.generate()
        self.assertTrue(torch.sum(torch.abs(ret_one - ret_two)) != 0.0)

    def test_normalize(self):
        generator = torch.Generator(torch.device("cpu"))
        generator.manual_seed(0)
        initializer = NormalInitializer(generator=generator)
        self.set_and_build_initializer(initializer)
        ret_one = initializer.generate()
        ret_two = initializer.generate()
        self.assertTrue(torch.sum(torch.abs(ret_one - ret_two)) != 0.0)

    def test_xavier_uniform(self):
        generator = torch.Generator(torch.device("cpu"))
        generator.manual_seed(0)
        initializer = XavierUniformInitializer(generator=generator)
        self.set_and_build_initializer(initializer)
        ret_one = initializer.generate()
        ret_two = initializer.generate()
        self.assertTrue(torch.sum(torch.abs(ret_one - ret_two)) != 0.0)

    def test_xavier_normal(self):
        generator = torch.Generator(torch.device("cpu"))
        generator.manual_seed(0)
        initializer = XavierNormalInitializer(generator=generator)
        self.set_and_build_initializer(initializer)
        ret_one = initializer.generate()
        ret_two = initializer.generate()
        self.assertTrue(torch.sum(torch.abs(ret_one - ret_two)) != 0.0)

    def test_kaiming_uniform(self):
        generator = torch.Generator(torch.device("cpu"))
        generator.manual_seed(0)
        initializer = KaimingUniformInitializer(generator=generator)
        self.set_and_build_initializer(initializer)
        ret_one = initializer.generate()
        ret_two = initializer.generate()
        self.assertTrue(torch.sum(torch.abs(ret_one - ret_two)) != 0.0)

    def test_kaiming_normal(self):
        generator = torch.Generator(torch.device("cpu"))
        generator.manual_seed(0)
        initializer = KaimingNormalInitializer(generator=generator)
        self.set_and_build_initializer(initializer)
        ret_one = initializer.generate()
        ret_two = initializer.generate()
        self.assertTrue(torch.sum(torch.abs(ret_one - ret_two)) != 0.0)

    def test_trunc_normal(self):
        generator = torch.Generator(torch.device("cpu"))
        generator.manual_seed(0)
        initializer = TruncNormalInitializer(generator=generator)
        self.set_and_build_initializer(initializer)
        ret_one = initializer.generate()
        ret_two = initializer.generate()
        self.assertTrue(torch.sum(torch.abs(ret_one - ret_two)) != 0.0)


if __name__ == "__main__":
    unittest.main()
