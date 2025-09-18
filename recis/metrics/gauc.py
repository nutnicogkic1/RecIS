import os

import torch
import torch.distributed as dist


class Gauc(torch.nn.Module):
    """Group AUC (GAUC) metric for recommendation systems and personalized ML tasks.

    This class computes the Group Area Under ROC Curve, which evaluates model
    performance by calculating AUC separately for each group (typically users)
    and then aggregating the results. This provides a more accurate assessment
    of recommendation system performance compared to global AUC metrics.

    GAUC is essential for recommendation systems because:
    - Different users have different behavior patterns and preferences
    - Global AUC can be dominated by users with many interactions
    - User-level evaluation provides better insights into model fairness
    - It's more aligned with business objectives in personalized systems

    The implementation uses optimized C++ operations for efficient computation
    and supports distributed training scenarios commonly found in large-scale
    recommendation systems.

    Attributes:
        _counts (float): Cumulative count of valid groups processed.
        _aucs (float): Cumulative weighted sum of AUC scores.
        _cpu (torch.device): CPU device for computation efficiency.
        _word_size (int): Number of distributed processes (world size).

    Example:
        Using GAUC in a recommendation model:

    .. code-block:: python


        # Initialize GAUC metric
        gauc_metric = Gauc()

        # Prepare recommendation data
        labels = torch.tensor([1, 0, 1, 0, 1, 0])  # Click labels
        predictions = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])  # CTR predictions
        user_ids = torch.tensor([1, 1, 2, 2, 3, 3])  # User identifiers

        # Compute GAUC
        batch_gauc, cumulative_gauc = gauc_metric(labels, predictions, user_ids)

        print(f"Batch GAUC: {batch_gauc:.4f}")
        print(f"Cumulative GAUC: {cumulative_gauc:.4f}")

        # Continue with more batches...
        # The cumulative GAUC will be updated automatically

        Integration with training loop:

    .. code-block:: python

        model = RecommendationModel()
        gauc_metric = Gauc()

        for epoch in range(num_epochs):
            gauc_metric.reset()  # Reset for new epoch

            for batch in train_dataloader:
                # Forward pass
                logits = model(batch)
                predictions = torch.sigmoid(logits)

                # Compute GAUC
                batch_gauc, epoch_gauc = gauc_metric(
                    batch["labels"], predictions, batch["user_ids"]
                )

                # Log metrics
                if batch_idx % 100 == 0:
                    print(
                        f"Epoch {epoch}, Batch {batch_idx}: "
                        f"Batch GAUC = {batch_gauc:.4f}, "
                        f"Epoch GAUC = {epoch_gauc:.4f}"
                    )

            print(f"Final Epoch {epoch} GAUC: {epoch_gauc:.4f}")
    """

    def __init__(self) -> None:
        """Initialize GAUC metric with default configuration.

        The metric automatically detects the distributed training environment
        and configures itself accordingly. It uses CPU computation for the
        GAUC calculation to optimize memory usage and computation efficiency.

        Note:
            The metric automatically reads the WORLD_SIZE environment variable
            to determine if running in distributed mode. In distributed training,
            it will aggregate results across all processes.
        """
        super().__init__()
        self._counts = 0.0
        self._aucs = 0.0
        self._cpu = torch.device("cpu")
        self._word_size = int(os.environ.get("WORLD_SIZE", 1))

    def reset(self):
        """Reset all accumulated statistics to zero.

        This method clears all internal state, resetting both the cumulative
        AUC sum and count statistics. It should be called at the beginning
        of each new evaluation period (e.g., new epoch) to ensure clean metrics.

        Example:

        .. code-block:: python

            gauc_metric = Gauc()

            for epoch in range(num_epochs):
                # Reset at the beginning of each epoch
                gauc_metric.reset()

                # Process batches for the epoch
                for batch in dataloader:
                    batch_gauc, epoch_gauc = gauc_metric(
                        batch["labels"], batch["predictions"], batch["user_ids"]
                    )

                print(f"Final Epoch {epoch} GAUC: {epoch_gauc:.4f}")

        """
        self._counts = 0.0
        self._aucs = 0.0

    def forward(self, labels, predictions, indicators):
        """Compute GAUC for the current batch and update cumulative statistics.

        This method computes the Group AUC by calculating AUC separately for each
        group identified by the indicators (typically user IDs) and then computing
        a weighted average. It returns both the current batch GAUC and the
        cumulative GAUC across all processed batches.

        Args:
            labels (torch.Tensor): Ground truth binary labels (0 or 1).
                Shape: (N,) where N is the number of samples.
            predictions (torch.Tensor): Predicted probabilities or scores.
                Shape: (N,) where N is the number of samples.
            indicators (torch.Tensor): Group identifiers (e.g., user IDs).
                Shape: (N,) where N is the number of samples.

        Returns:
            tuple: A tuple containing:
                - batch_gauc (float): GAUC score for the current batch
                - cumulative_gauc (float): Cumulative GAUC across all processed batches

        Example:

        .. code-block:: python

            gauc_metric = Gauc()

            # Single batch computation
            labels = torch.tensor([1, 0, 1, 0, 1, 0])
            predictions = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
            user_ids = torch.tensor([1, 1, 2, 2, 3, 3])

            batch_gauc, cumulative_gauc = gauc_metric(labels, predictions, user_ids)
            print(f"Batch GAUC: {batch_gauc:.4f}")
            print(f"Cumulative GAUC: {cumulative_gauc:.4f}")

            # Process another batch
            labels2 = torch.tensor([0, 1, 1, 0])
            predictions2 = torch.tensor([0.2, 0.8, 0.9, 0.1])
            user_ids2 = torch.tensor([4, 4, 5, 5])

            batch_gauc2, cumulative_gauc2 = gauc_metric(
                labels2, predictions2, user_ids2
            )
            print(f"Batch 2 GAUC: {batch_gauc2:.4f}")
            print(f"Updated Cumulative GAUC: {cumulative_gauc2:.4f}")


        Note:
            The method automatically handles distributed training by aggregating
            results across all processes when WORLD_SIZE > 1. The computation
            is performed on CPU for memory efficiency, with automatic device
            transfer handled internally.

            The GAUC calculation weights each group's AUC by the number of
            samples in that group, providing a fair aggregation across groups
            of different sizes.
        """
        with torch.no_grad():
            aucs, counts = torch.ops.recis.gauc_calc(
                labels.to(self._cpu),
                predictions.to(self._cpu),
                indicators.to(self._cpu),
            )
            aucs = aucs * counts
            sum_aucs = torch.sum(aucs)
            sum_counts = torch.sum(counts)
            if self._word_size != 1:
                reduce_val = torch.stack([sum_aucs, sum_counts]).to(labels.device)
                dist.all_reduce(reduce_val)
            else:
                reduce_val = torch.stack([sum_aucs, sum_counts])
            split_sum_auc, split_sum_count = torch.split(
                reduce_val, split_size_or_sections=1
            )
            split_sum_auc_val = split_sum_auc.item()
            split_sum_count_val = split_sum_count.item()
            self._counts += split_sum_count_val
            self._aucs += split_sum_auc_val

            batch_gauc = split_sum_auc_val / max(split_sum_count_val, 1.0)
            cumulative_gauc = self._aucs / max(self._counts, 1.0)

            return batch_gauc, cumulative_gauc
