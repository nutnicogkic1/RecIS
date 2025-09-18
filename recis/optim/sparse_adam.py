import torch

from recis.optim.sparse_optim import SparseOptimizer


class SparseAdam(SparseOptimizer):
    """Sparse Adam optimizer for efficient sparse parameter optimization.

    This class implements the Adam optimization algorithm specifically optimized
    for sparse parameters in recommendation systems. It extends the SparseOptimizer
    base class and uses RecIS's C++ implementation for maximum performance.

    The Adam algorithm uses adaptive learning rates computed from estimates of
    first and second moments of gradients. For sparse parameters, this implementation
    only updates parameters that have received gradients, making it highly efficient
    for large embedding tables where only a small fraction of parameters are
    active in each training step.

    Key advantages for sparse parameters:
    - Only active parameters are updated, saving computation
    - Separate adaptive learning rates for each sparse parameter
    - Efficient memory usage for momentum and variance estimates
    - Optimized C++ implementation for maximum performance

    Example:
        Creating and using SparseAdam:

    .. code-block:: python

        # Initialize with default hyperparameters
        optimizer = SparseAdam(
            param_dict=sparse_parameters,
            lr=0.001,  # Learning rate
            beta1=0.9,  # First moment decay rate
            beta2=0.999,  # Second moment decay rate
            eps=1e-8,  # Numerical stability
            weight_decay=0.0,  # No weight decay by default
        )

        # Training with gradient accumulation
        optimizer.set_grad_accum_steps(4)

        for batch in dataloader:
            loss = model(batch) / 4  # Scale for accumulation
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    """

    def __init__(
        self,
        param_dict: dict,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
    ) -> None:
        """Initialize SparseAdam optimizer with specified hyperparameters.

        Args:
            param_dict (dict): Dictionary of sparse parameters to optimize.
                Keys are parameter names, values are parameter tensors (typically HashTables).
            lr (float, optional): Learning rate. Defaults to 1e-3.
            beta1 (float, optional): Exponential decay rate for first moment estimates.
                Should be in [0, 1). Defaults to 0.9.
            beta2 (float, optional): Exponential decay rate for second moment estimates.
                Should be in [0, 1). Defaults to 0.999.
            eps (float, optional): Small constant added to denominator for numerical
                stability. Defaults to 1e-8.
            weight_decay (float, optional): Weight decay coefficient. Note that unlike
                SparseAdamW, this applies L2 penalty to gradients rather than direct
                weight decay. Defaults to 1e-2.
            amsgrad (bool, optional): Whether to use AMSGrad variant. Currently not
                supported and will raise ValueError if True. Defaults to False.

        Raises:
            ValueError: If amsgrad is True (not currently supported).

        Note:
            The param_dict should contain HashTable parameters that support
            sparse gradient updates. The weight_decay in SparseAdam applies
            L2 penalty to gradients, which is different from the direct weight
            decay used in SparseAdamW.
        """
        super().__init__(lr=lr)

        # Store hyperparameters
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._weight_decay = weight_decay

        # AMSGrad is not currently supported
        if amsgrad:
            raise ValueError("amsgrad is not support now")
        self._amsgrad = amsgrad

        # Create the underlying C++ optimizer implementation
        self._imp = torch.classes.recis.SparseAdam.make(
            param_dict,
            self._lr,
            self._beta1,
            self._beta2,
            self._eps,
            self._weight_decay,
            self._amsgrad,
        )
