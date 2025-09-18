Evaluation Metrics Module
=========================

RecIS's evaluation metrics module provides commonly used evaluation metrics for recommendation systems and machine learning, supporting distributed computing and real-time updates.

Core Metrics
------------

AUROC
~~~~~

.. currentmodule:: recis.metrics.auroc

.. autoclass:: AUROC
    :members: __init__, update, compute, reset, forward
      

GAUC
~~~~

.. currentmodule:: recis.metrics.gauc

.. autoclass:: Gauc
    :members: __init__, forward, reset

Metrics Integration
-------------------

Integrating Metrics in Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RecIS provides convenient ways to integrate evaluation metrics in model training:

.. code-block:: python

   import torch.nn as nn
   from recis.metrics import AUROC, GAUC
   from recis.framework.metrics import add_metric
   
   class RecommendationModel(nn.Module):
       def __init__(self):
           super().__init__()
           # Model components
           self.embedding = ...
           self.dnn = ...
           
           # Evaluation metrics
           self.auc_metric = AUROC(num_thresholds=200, dist_sync_on_step=True)
           self.gauc_metric = GAUC(num_thresholds=200)
           
           self.loss_fn = nn.BCELoss()
       
       def forward(self, batch):
           # Model forward pass
           logits = self.predict(batch)
           labels = batch['label']
           user_ids = batch['user_id']
           
           # Compute loss
           loss = self.loss_fn(logits, labels.float())
           
           # Update metrics
           self.auc_metric.update(logits, labels)
           self.gauc_metric.update(logits, labels, user_ids)
           
           # Compute metric values
           auc = self.auc_metric.compute()
           gauc = self.gauc_metric.compute()
           
           # Add to training framework's metric system
           add_metric("auc", auc)
           add_metric("gauc", gauc)
           add_metric("loss", loss)
           
           return loss

Distributed Metrics Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using metrics in distributed training:

.. code-block:: python

   import torch.distributed as dist
   from recis.metrics import AUROC
   
   # Ensure distributed environment is initialized
   if dist.is_initialized():
       # Enable distributed synchronization
       auc_metric = AUROC(
           num_thresholds=200,
           dist_sync_on_step=True  # Sync at each step for consistency
       )
   else:
       auc_metric = AUROC(num_thresholds=200)
   
   # Use normally in training loop
   for batch in dataloader:
       preds = model(batch)
       labels = batch['label']
       
       # Metrics will automatically handle distributed aggregation
       auc_metric.update(preds, labels)
       auc_score = auc_metric.compute()

Custom Metrics
--------------

Creating custom evaluation metrics:

.. code-block:: python

   import torch
   from typing import Any, Optional
   
   class CustomMetric:
       def __init__(self, dist_sync_on_step: bool = False):
           self.dist_sync_on_step = dist_sync_on_step
           self.reset()
       
       def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
           """Update metric state"""
           # Implement metric update logic
           self.total_samples += preds.size(0)
           self.correct_predictions += (preds.round() == target).sum()
       
       def compute(self) -> torch.Tensor:
           """Compute metric value"""
           if self.total_samples == 0:
               return torch.tensor(0.0)
           
           accuracy = self.correct_predictions.float() / self.total_samples
           
           # Distributed synchronization
           if self.dist_sync_on_step and dist.is_initialized():
               dist.all_reduce(accuracy, op=dist.ReduceOp.AVG)
           
           return accuracy
       
       def reset(self) -> None:
           """Reset metric state"""
           self.total_samples = 0
           self.correct_predictions = 0

   # Use custom metric
   custom_metric = CustomMetric(dist_sync_on_step=True)

Frequently Asked Questions
--------------------------

**Q: How to correctly use metrics in distributed training?**

A: Ensure correct synchronization parameters are set:

.. code-block:: python

   # During training: sync at each step for consistency
   train_auc = AUROC(num_thresholds=200, dist_sync_on_step=True)
   
   # During validation: sync at the end is sufficient
   val_auc = AUROC(num_thresholds=200, dist_sync_on_step=False)

**Q: What's the difference between GAUC and AUC?**

A: 
- AUC: Global computation, all samples together to compute ROC curve
- GAUC: Grouped computation, first compute AUC by groups (e.g., users), then weighted average

**Q: How to save and load metric states?**

A: Metric objects support state saving:

.. code-block:: python

   # Save metric state
   metric_state = auc_metric.state_dict()
   torch.save(metric_state, 'metric_state.pth')
   
   # Load metric state
   auc_metric = AUROC(num_thresholds=200)
   metric_state = torch.load('metric_state.pth')
   auc_metric.load_state_dict(metric_state)