import torch
import torch.distributed as dist
from torch import nn


class AUROC(nn.Module):
    """Area Under ROC Curve (AUROC) metric for binary classification.

    This class computes the Area Under the Receiver Operating Characteristic (ROC) Curve,
    which measures the ability of a binary classifier to distinguish between positive and
    negative classes across all classification thresholds. The implementation uses confusion
    matrices computed at multiple thresholds for efficient and accurate AUC calculation.

    The AUROC metric is particularly useful for:
    - Binary classification tasks in recommendation systems
    - Click-through rate (CTR) prediction
    - Conversion rate optimization
    - Any binary classification where class balance matters

    Attributes:
        num_thresholds (int): Number of thresholds used for ROC curve computation.
        dist_sync_on_step (bool): Whether to synchronize across devices on each update.
        thresholds (torch.Tensor): Threshold values used for classification decisions.
        tp (nn.Parameter): True positive counts at each threshold.
        fp (nn.Parameter): False positive counts at each threshold.
        tn (nn.Parameter): True negative counts at each threshold.
        fn (nn.Parameter): False negative counts at each threshold.

    Example:
        Creating and using AUROC metric:

    .. code-block:: python

        # Initialize with custom configuration
        auc_metric = AUROC(
            num_thresholds=100,  # Use 100 thresholds for ROC curve
            dist_sync_on_step=False,  # Sync only when computing final result
        )

        # Batch processing
        predictions = torch.tensor([0.9, 0.7, 0.3, 0.1])
        labels = torch.tensor([1, 1, 0, 0])

        # Update metric state
        auc_metric.update(predictions, labels)

        # Compute AUC
        auc_score = auc_metric.compute()
        print(f"AUC: {auc_score:.4f}")

        # Direct computation (alternative to update + compute)
        direct_auc = auc_metric(predictions, labels)

    """

    def __init__(self, num_thresholds=200, dist_sync_on_step=False):
        """Initialize AUROC metric with specified configuration.

        Args:
            num_thresholds (int, optional): Number of thresholds to use for ROC curve
                computation. Must be greater than 2. Defaults to 200.
            dist_sync_on_step (bool, optional): Whether to synchronize metric state
                across distributed processes on each update step. If False, synchronization
                only occurs during compute(). Defaults to False.

        Raises:
            AssertionError: If num_thresholds is not greater than 2.

        Note:
            Higher num_thresholds values provide more accurate AUC computation but
            require more memory and computation. The thresholds are evenly distributed
            between 0 and 1 with small epsilon values at the boundaries.
        """
        super().__init__()
        assert num_thresholds > 2, "num_thresholds must be > 2"
        self.num_thresholds = num_thresholds
        self.dist_sync_on_step = dist_sync_on_step

        # Small epsilon to handle boundary cases
        kepsilon = 1e-7

        # Create evenly spaced thresholds between 0 and 1
        thresholds = [
            (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
        ]
        self.thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]
        self.thresholds = torch.tensor(self.thresholds)

        # Initialize confusion matrix components as parameters
        self.tp = nn.Parameter(
            torch.zeros(num_thresholds, dtype=torch.long), requires_grad=False
        )
        self.fp = nn.Parameter(
            torch.zeros(num_thresholds, dtype=torch.long), requires_grad=False
        )
        self.tn = nn.Parameter(
            torch.zeros(num_thresholds, dtype=torch.long), requires_grad=False
        )
        self.fn = nn.Parameter(
            torch.zeros(num_thresholds, dtype=torch.long), requires_grad=False
        )

    def _confusion_matrix_at_thresholds(self, predictions, labels):
        """Compute confusion matrix components at all thresholds.

        This method efficiently computes true positives, false positives, true negatives,
        and false negatives for all configured thresholds simultaneously using vectorized
        operations.

        Args:
            predictions (torch.Tensor): Predicted probabilities in range [0, 1].
                Shape: (N,) where N is the number of samples.
            labels (torch.Tensor): Ground truth binary labels (0 or 1).
                Shape: (N,) where N is the number of samples.

        Returns:
            tuple: A tuple containing:
                - tp (torch.Tensor): True positive counts for each threshold. Shape: (num_thresholds,)
                - fp (torch.Tensor): False positive counts for each threshold. Shape: (num_thresholds,)
                - tn (torch.Tensor): True negative counts for each threshold. Shape: (num_thresholds,)
                - fn (torch.Tensor): False negative counts for each threshold. Shape: (num_thresholds,)

        Raises:
            AssertionError: If predictions are not in the range [0, 1].

        Note:
            This method uses efficient tensor operations to compute confusion matrices
            for all thresholds simultaneously, avoiding expensive loops.
        """
        assert torch.all(torch.logical_and(predictions >= 0.0, predictions <= 1.0)), (
            "predictions must be in [0, 1]"
        )

        predictions_1d = predictions.view(-1)
        labels_1d = labels.to(dtype=torch.bool).view(-1)
        self.thresholds = self.thresholds.to(predictions.device)

        # Compute predictions > threshold for all thresholds
        pred_is_pos = predictions_1d.unsqueeze(-1) > self.thresholds

        # Transpose to get shape (num_thresholds, num_samples)
        pred_is_pos = pred_is_pos.t()
        pred_is_neg = torch.logical_not(pred_is_pos)
        label_is_pos = labels_1d.repeat(self.num_thresholds, 1)
        label_is_neg = torch.logical_not(label_is_pos)

        # Compute confusion matrix components
        is_true_positive = torch.logical_and(label_is_pos, pred_is_pos)
        is_true_negative = torch.logical_and(label_is_neg, pred_is_neg)
        is_false_positive = torch.logical_and(label_is_neg, pred_is_pos)
        is_false_negative = torch.logical_and(label_is_pos, pred_is_neg)

        # Sum across samples for each threshold
        tp = is_true_positive.sum(1)
        fn = is_false_negative.sum(1)
        tn = is_true_negative.sum(1)
        fp = is_false_positive.sum(1)

        return tp, fp, tn, fn

    def _compute_auroc(self, tp, fp, tn, fn):
        """Compute AUROC from confusion matrix components.

        This method calculates the Area Under the ROC Curve using the trapezoidal rule
        for numerical integration. The ROC curve is defined by true positive rate (TPR)
        vs false positive rate (FPR) at different thresholds.

        Args:
            tp (torch.Tensor): True positive counts for each threshold.
            fp (torch.Tensor): False positive counts for each threshold.
            tn (torch.Tensor): True negative counts for each threshold.
            fn (torch.Tensor): False negative counts for each threshold.

        Returns:
            torch.Tensor: AUROC score as a scalar tensor.

        Note:
            Uses small epsilon values to prevent division by zero and ensure
            numerical stability. The trapezoidal rule provides accurate AUC
            approximation when sufficient thresholds are used.
        """
        epsilon = 1.0e-6

        # Compute True Positive Rate (Recall/Sensitivity)
        rec = torch.div(tp + epsilon, tp + fn + epsilon)

        # Compute False Positive Rate (1 - Specificity)
        fp_rate = torch.div(fp, fp + tn + epsilon)

        x = fp_rate
        y = rec

        # Compute AUC using trapezoidal rule
        auc = torch.multiply(
            x[: self.num_thresholds - 1] - x[1:],
            (y[: self.num_thresholds - 1] + y[1:]) / 2.0,
        ).sum()

        return auc

    def forward(self, predictions, labels):
        """Compute AUROC directly from predictions and labels.

        This method provides a direct way to compute AUROC without updating
        the internal state. It's useful for one-time computations or when
        you don't need to accumulate statistics across multiple batches.

        Args:
            predictions (torch.Tensor): Predicted probabilities in range [0, 1].
                Shape: (N,) where N is the number of samples.
            labels (torch.Tensor): Ground truth binary labels (0 or 1).
                Shape: (N,) where N is the number of samples.

        Returns:
            torch.Tensor: AUROC score as a scalar tensor.

        Example:

        .. code-block:: python

            auc_metric = AUROC(num_thresholds=100)

            # Direct computation
            predictions = torch.tensor([0.9, 0.7, 0.3, 0.1])
            labels = torch.tensor([1, 1, 0, 0])
            auc_score = auc_metric(predictions, labels)
            print(f"AUC: {auc_score:.4f}")

        """
        tp, fp, tn, fn = self._confusion_matrix_at_thresholds(predictions, labels)
        return self._compute_auroc(tp, fp, tn, fn)

    def update(self, predictions, labels):
        """Update metric state with new predictions and labels.

        This method accumulates confusion matrix statistics from the current batch
        with previously seen data. It's designed for incremental updates during
        training where you want to compute metrics across multiple batches.

        Args:
            predictions (torch.Tensor): Predicted probabilities in range [0, 1].
                Shape: (N,) where N is the number of samples.
            labels (torch.Tensor): Ground truth binary labels (0 or 1).
                Shape: (N,) where N is the number of samples.

        Example:

        .. code-block:: python

            auc_metric = AUROC(num_thresholds=200, dist_sync_on_step=True)

            # Process multiple batches
            for batch in dataloader:
                preds = model(batch)
                labels = batch["labels"]

                # Accumulate statistics
                auc_metric.update(preds, labels)

            # Get final result
            final_auc = auc_metric.compute()


        Note:
            If dist_sync_on_step is True, this method will synchronize statistics
            across all distributed processes, which may impact performance but
            ensures consistency in distributed training.
        """
        tp, fp, tn, fn = self._confusion_matrix_at_thresholds(predictions, labels)

        # Synchronize across distributed processes if required
        if self.dist_sync_on_step:
            tp, fp, tn, fn = self.sync(tp, fp, tn, fn)

        # Accumulate statistics
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def sync(self, tp, fp, tn, fn):
        """Synchronize confusion matrix statistics across distributed processes.

        This method aggregates confusion matrix components from all distributed
        processes using all-reduce operations. It's essential for consistent
        metric computation in distributed training scenarios.

        Args:
            tp (torch.Tensor): True positive counts to synchronize.
            fp (torch.Tensor): False positive counts to synchronize.
            tn (torch.Tensor): True negative counts to synchronize.
            fn (torch.Tensor): False negative counts to synchronize.

        Returns:
            tuple: Synchronized confusion matrix components:
                - tp (torch.Tensor): Synchronized true positive counts
                - fp (torch.Tensor): Synchronized false positive counts
                - tn (torch.Tensor): Synchronized true negative counts
                - fn (torch.Tensor): Synchronized false negative counts

        Note:
            This method requires PyTorch distributed training to be properly
            initialized. It uses SUM reduction to aggregate counts across processes.
        """
        # Concatenate all statistics for efficient communication
        state = torch.cat([tp, fp, tn, fn], dim=0)

        # Perform all-reduce sum across all processes
        dist.all_reduce(state, op=dist.ReduceOp.SUM)

        # Split back into individual components
        tp, fp, tn, fn = state.split(
            [self.tp.numel(), self.fp.numel(), self.tn.numel(), self.fn.numel()], dim=0
        )

        return tp, fp, tn, fn

    def compute(self):
        """Compute final AUROC score from accumulated statistics.

        This method calculates the AUROC using all statistics accumulated through
        previous update() calls. It's typically called at the end of an epoch or
        evaluation period to get the final metric value.

        Returns:
            torch.Tensor: AUROC score as a scalar tensor.

        Example:

        .. code-block:: python

            auc_metric = AUROC()

            # Accumulate data from multiple batches
            for batch in dataloader:
                auc_metric.update(model(batch), batch["labels"])

            # Get final AUC score
            final_auc = auc_metric.compute()
            print(f"Epoch AUC: {final_auc:.4f}")


        Note:
            This method uses the current accumulated state (tp, fp, tn, fn) to
            compute the final AUROC. Make sure to call reset() before starting
            a new evaluation period.
        """
        return self._compute_auroc(self.tp, self.fp, self.tn, self.fn)

    def reset(self):
        """Reset all accumulated statistics to zero.

        This method clears all internal state, setting all confusion matrix
        components back to zero. It should be called at the beginning of each
        new evaluation period (e.g., new epoch) to ensure clean statistics.

        Example:

        .. code-block:: python

            auc_metric = AUROC()

            for epoch in range(num_epochs):
                # Reset at the beginning of each epoch
                auc_metric.reset()

                # Accumulate statistics for the epoch
                for batch in dataloader:
                    auc_metric.update(model(batch), batch["labels"])

                # Get epoch result
                epoch_auc = auc_metric.compute()
                print(f"Epoch {epoch} AUC: {epoch_auc:.4f}")


        Note:
            This method modifies the internal parameter tensors in-place using
            zero_() for efficiency.
        """
        self.tp.zero_()
        self.fp.zero_()
        self.tn.zero_()
        self.fn.zero_()
