import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import get_embedding_conf, get_feature_conf
from feature_config import FEATURE_CONFIG
from transformer import ModelConfig, Transformer

from recis.features.feature_engine import FeatureEngine
from recis.framework.metrics import add_metric
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
        return samples


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.proj = nn.Linear(config.emb_size, config.hidden_size)
        self.trans = Transformer(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def cal_loss(
        self,
        preds: torch.Tensor,
        items: torch.Tensor,
    ):
        preds = preds[:, 1:, :]
        preds = preds.reshape(-1, preds.shape[-1])
        items = items[:, :-1, :]
        items = items.reshape(-1, items.shape[-1])
        preds = F.normalize(preds, p=2, dim=-1, eps=1e-6)
        items = F.normalize(items, p=2, dim=-1, eps=1e-6)
        labels = torch.arange(preds.shape[0], device=preds.device, dtype=torch.long)
        cos_sim = torch.matmul(preds, items.t())
        loss = self.loss_fn(cos_sim, labels)
        with torch.no_grad():
            hits = (cos_sim.detach().argmax(dim=1) == labels).sum()
        add_metric("hit_rate", hits / preds.shape[0])
        add_metric("loss", loss)
        return loss

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        items = self.proj(x)
        preds = self.trans(items, attn_mask)
        if self.training:
            loss = self.cal_loss(preds, items)
            return loss
        else:
            return preds


class Seq2SeqModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = Decoder(config)
        self.sparse = SparseModel()
        self.casual_mask = (
            torch.tril(torch.ones(config.seq_len, config.seq_len))
            .view(1, 1, config.seq_len, config.seq_len)
            .cuda()
        )

    def build_embedding(self, samples: dict[torch.Tensor]):
        embs = []
        for item in FEATURE_CONFIG:
            fn = item["name"]
            embs.append(samples[fn])
        return torch.cat(embs, dim=-1)

    def cal_mask(self, seq_len):
        return self.casual_mask[:, :, :seq_len, :seq_len]

    def forward(self, samples: dict[torch.Tensor]):
        samples = self.sparse(samples)
        emb = self.build_embedding(samples)
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            return self.dense(emb, self.cal_mask(emb.shape[1]))
