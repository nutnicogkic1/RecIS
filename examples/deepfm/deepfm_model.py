import torch
import torch.nn as nn
from dataset import get_embedding_conf, get_feature_conf
from feature_config import DNN_SHAPE, EMBEDDING_DIM, FEATURES

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
        self.embedding_dim = EMBEDDING_DIM
        num_numeric_features = len(FEATURES["numeric"])
        num_categorical_features = len(FEATURES["categorical"])

        # Initialize weights and biases for numerical features
        self.numeric_embs = nn.Parameter(
            torch.randn(num_numeric_features, self.embedding_dim) * 0.001
        )
        self.numeric_biases = nn.Parameter(torch.randn(num_numeric_features, 1) * 0.001)

        # Build the DNN part
        layers = []
        dnn_structure = DNN_SHAPE
        input_size = (
            num_numeric_features + num_categorical_features
        ) * self.embedding_dim
        for size in dnn_structure:
            layers.extend([nn.Linear(input_size, size), nn.ReLU()])
            input_size = size
        layers.append(nn.Linear(input_size, 1))  # Output layer
        self.dnn = nn.Sequential(*layers)

    def forward(self, samples, labels):
        batch_size = labels.shape[0]

        # Get numeric features
        numeric_embeddings = self.numeric_embs.repeat(batch_size, 1, 1)
        numeric_biases = self.numeric_biases.repeat(batch_size, 1, 1)
        numeric_weights = torch.stack(
            [samples[fn] for fn in FEATURES["numeric"]],
            dim=1,
        )

        # Get embeddings and biases for categorical features
        category_embeddings = torch.stack(
            [samples[f"{fn}_emb"] for fn in FEATURES["categorical"]],
            dim=1,
        )
        category_biases = torch.stack(
            [samples[f"{fn}_bias"] for fn in FEATURES["categorical"]],
            dim=1,
        )
        category_weights = torch.ones(
            [batch_size, len(FEATURES["categorical"]), 1], device=category_biases.device
        )

        # Merge all feature embeddings and biases
        all_embeddings = torch.cat([numeric_embeddings, category_embeddings], dim=1)
        all_biases = torch.cat([numeric_biases, category_biases], dim=1)
        all_weights = torch.cat([numeric_weights, category_weights], dim=1)

        # Calculate first-order effects
        first_order_output = torch.sum(
            torch.squeeze(all_weights * all_biases, dim=-1), dim=-1, keepdim=True
        )

        # Calculate second-order effects
        squared_sum = torch.sum(all_embeddings * all_weights, dim=1) ** 2
        sum_squared = torch.sum((all_embeddings**2) * (all_weights**2), dim=1)
        second_order_output = 0.5 * torch.sum(
            squared_sum - sum_squared, dim=-1, keepdim=True
        )

        # DNN output
        dnn_input = all_embeddings.view(
            -1, all_embeddings.shape[1] * all_embeddings.shape[2]
        )
        dnn_output = self.dnn(dnn_input)

        # Final output
        final_output = torch.sigmoid(
            first_order_output + second_order_output + dnn_output
        )
        return final_output


class DeepFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.sparse_model = SparseModel()
        self.dense_model = DenseModel()
        self.loss_function = nn.BCELoss()
        self.auc_metric = AUROC(num_thresholds=200, dist_sync_on_step=True)

    def forward(self, samples):
        samples, labels = self.sparse_model(samples)
        final_output = self.dense_model(samples, labels)
        # Calculate loss
        loss = self.loss_function(final_output, labels)

        self.auc_metric.update(final_output, labels)
        auc = self.auc_metric.compute()
        add_metric("auc", auc)

        return loss


#
