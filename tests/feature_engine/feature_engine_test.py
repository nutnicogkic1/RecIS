import unittest

import torch

from recis.features.feature import Feature
from recis.features.feature_engine import FeatureEngine
from recis.features.op import (
    Bucketize,
    FeatureCross,
    Mod,
    SelectField,
    SelectFields,
    SequenceTruncate,
)
from recis.ragged.tensor import RaggedTensor


def make_fake_data():
    data = {
        "item_id": RaggedTensor(
            values=torch.LongTensor([1, 2, 3]),
            offsets=torch.IntTensor([0, 1, 2, 3]),
        ).cuda(),
        "user_id": RaggedTensor(
            values=torch.LongTensor([1, 2, 3]),
            offsets=torch.IntTensor([0, 1, 2, 3]),
        ).cuda(),
        "price": RaggedTensor(
            values=torch.Tensor([0.5, 10, 11.6]).float(),
            offsets=torch.IntTensor([0, 1, 2, 3]),
        ).cuda(),
        "seq_item_id": RaggedTensor(
            values=torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8]),
            offsets=[torch.IntTensor(0, 1, 2), torch.IntTensor([0, 5, 8])],
        ).cuda(),
        "x": RaggedTensor(
            values=torch.LongTensor([1, 2, 3]),
            offsets=torch.IntTensor([0, 1, 2, 3]),
            weight=torch.ones([3]),
        ).cuda(),
        "y": RaggedTensor(
            values=torch.LongTensor([1, 2, 3]),
            offsets=torch.IntTensor([0, 1, 2, 3]),
            weight=torch.ones([3]),
        ).cuda(),
    }
    return data


def define_features():
    item_id = Feature("item_id").add_op(SelectField("item_id")).add_op(Mod(1000000007))
    sitem_id = (
        Feature("item_id_2").add_op(SelectField("item_id")).add_op(Mod(1000000007))
    )
    user_id = Feature("user_id").add_op(SelectField("user_id")).add_op(Mod(1000000007))

    price = (
        Feature("price").add_op(SelectField("price")).add_op(Bucketize([0, 100, 200]))
    )
    seq_item_id = (
        Feature("seq_item_id")
        .add_op(SelectField("seq_item_id"))
        .add_op(SequenceTruncate(seq_len=16, truncate=True, check_length=False))
        .add_op(Mod(1000000007))
    )
    combo_fn = (
        Feature("combo")
        .add_op(SelectFields([SelectField("x"), SelectField("y")]))
        .add_op(FeatureCross())
        .add_op(Mod(1000))
    )

    return [item_id, sitem_id, user_id, price, seq_item_id, combo_fn]


def get_result():
    data = {
        "item_id": RaggedTensor(
            values=torch.Tensor([1, 2, 3]),
            offsets=[torch.Tensor([0, 1, 2, 3]).int()],
        ),
        "item_id_2": RaggedTensor(
            values=torch.Tensor([1, 2, 3]),
            offsets=[torch.Tensor([0, 1, 2, 3]).int()],
        ),
        "user_id": RaggedTensor(
            values=torch.Tensor([1, 2, 3]),
            offsets=[torch.Tensor([0, 1, 2, 3]).int()],
        ),
        "price": RaggedTensor(
            values=torch.Tensor([1, 1, 1]),
            offsets=[torch.Tensor([0, 1, 2, 3]).int()],
        ),
        "seq_item_id": RaggedTensor(
            values=torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            offsets=[torch.Tensor([0, 5, 8]).int()],
        ),
        "combo": RaggedTensor(
            values=torch.Tensor([216, 2, 783]),
            offsets=[torch.Tensor([0, 1, 2, 3]).int()],
            weight=torch.Tensor([1.0, 1.0, 1.0]),
        ),
    }
    return data


def ragged_eq(a, b):
    x = torch.equal(a.values().cpu(), b.values().cpu())
    x = x and len(a.offsets()) == len(b.offsets())
    for i in range(len(a.offsets())):
        x = x and torch.equal(a.offsets()[i].cpu(), b.offsets()[i].cpu())
    weight_count = (a.weight() is not None) + (b.weight() is not None)
    if weight_count == 1:
        return False
    if weight_count == 2:
        x = x and torch.equal(a.weight().cpu(), b.weight().cpu())
    return x


class FeatureTest(unittest.TestCase):
    def test_feature(self):
        features = define_features()
        data = make_fake_data()
        ans = get_result()
        for feature in features:
            output = feature(data)
            self.assertTrue(ragged_eq(output, ans[feature.name]))


class FeatureEngineTest(unittest.TestCase):
    def test_feature_engine(self):
        features = define_features()
        feature_engine = FeatureEngine(feature_list=features)
        print(feature_engine)
        result = feature_engine(make_fake_data(), remain_no_use_data=False)
        ans = get_result()
        for key in result:
            self.assertTrue(ragged_eq(result[key], ans[key]))


if __name__ == "__main__":
    unittest.main()
