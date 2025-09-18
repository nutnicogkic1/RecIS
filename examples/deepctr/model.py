import torch
import torch.nn as nn
from dataset import get_embedding_conf, get_feature_conf

from recis.features.feature_engine import FeatureEngine
from recis.framework.metrics import add_metric
from recis.metrics.auroc import AUROC
from recis.nn import EmbeddingEngine
from recis.utils.logger import Logger


logger = Logger(__name__)


class SparseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_engine = FeatureEngine(feature_list=get_feature_conf())
        self.embedding_engine = EmbeddingEngine(get_embedding_conf())

    def forward(self, samples: dict):
        samples = self.feature_engine(samples)
        samples = self.embedding_engine(samples)
        labels = samples.pop("label")
        return samples, labels


class DenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        layers.extend(
            [nn.Linear(8 + 8 + 8 + 16 + 8 + 16000 + 16000 + 16, 128), nn.ReLU()]
        )
        layers.extend([nn.Linear(128, 64), nn.ReLU()])
        layers.extend([nn.Linear(64, 32), nn.ReLU()])
        layers.extend([nn.Linear(32, 1)])
        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.dnn(x)
        logits = torch.sigmoid(x)
        return logits


class DeepCTR(nn.Module):
    def __init__(self):
        super().__init__()
        self.sparse_arch = SparseModel()
        self.dense_arch = DenseModel()
        self.auc_metric = AUROC(num_thresholds=200, dist_sync_on_step=True)
        self.loss = nn.BCELoss()

    def forward(self, samples: dict):
        samples, labels = self.sparse_arch(samples)
        dense_input = torch.cat(
            [
                samples["dense1"],
                samples["dense2"],
                samples["sparse1"],
                samples["sparse2"],
                samples["sparse3"],
                samples["sparse4"].view(samples["sparse4"].shape[0], -1),
                samples["sparse5"].view(samples["sparse5"].shape[0], -1),
                samples["sparse1_x_sparse2"],
            ],
            -1,
        )
        logits = self.dense_arch(dense_input)
        loss = self.loss(logits, labels)
        self.auc_metric.update(logits, labels)
        auc = self.auc_metric.compute()
        add_metric("auc", auc)
        return loss
