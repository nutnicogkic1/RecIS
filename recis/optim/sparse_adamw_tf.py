import torch

from recis.optim.sparse_optim import SparseOptimizer


class SparseAdamWTF(SparseOptimizer):
    """Sparse AdamW optimizer with TensorFlow-style implementation for efficient sparse parameter optimization.

    This class implements the AdamW optimization algorithm with TensorFlow-compatible
    behavior specifically optimized for sparse parameters in recommendation systems.
    It extends the SparseOptimizer base class and uses RecIS's C++ implementation
    for maximum performance.

    Example:
        Creating and using SparseAdamWTF:

    .. code-block:: python

        # Initialize with custom hyperparameters for TF compatibility
        optimizer = SparseAdamWTF(
            param_dict=sparse_parameters,
            lr=0.001,  # Learning rate
            beta1=0.9,  # First moment decay rate
            beta2=0.999,  # Second moment decay rate
            eps=1e-8,  # Numerical stability
            weight_decay=0.01,  # L2 regularization strength
            use_nesterov=False,  # Nesterov momentum (not supported yet)
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
        use_nesterov=False,
    ) -> None:
        """Initialize SparseAdamWTF optimizer with specified hyperparameters.

        Args:
            param_dict (dict): Dictionary of sparse parameters to optimize.
                Keys are parameter names, values are parameter tensors (typically HashTables).
            lr (float, optional): Learning rate. Should match TensorFlow training
                settings for compatibility. Defaults to 1e-3.
            beta1 (float, optional): Exponential decay rate for first moment estimates.
                Should be in [0, 1). TensorFlow default is 0.9. Defaults to 0.9.
            beta2 (float, optional): Exponential decay rate for second moment estimates.
                Should be in [0, 1). TensorFlow default is 0.999. Defaults to 0.999.
            eps (float, optional): Small constant added to denominator for numerical
                stability. TensorFlow default is 1e-7, but 1e-8 is also common.
                Defaults to 1e-8.
            weight_decay (float, optional): Weight decay coefficient (L2 regularization).
                Applied directly to parameters (decoupled weight decay). Defaults to 1e-2.
            use_nesterov (bool, optional): Whether to use Nesterov momentum variant.
                Currently not supported and will raise ValueError if True. Defaults to False.

        Raises:
            ValueError: If use_nesterov is True (not currently supported).

        Note:
            The param_dict should contain HashTable parameters that support
            sparse gradient updates. This optimizer is specifically designed
            for compatibility with TensorFlow-trained models and provides
            numerically equivalent behavior for seamless model migration.
        """
        super().__init__(lr=lr)
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._weight_decay = weight_decay
        if use_nesterov:
            raise ValueError("use_nesterov is not support now")
        self._use_nesterov = use_nesterov
        self._imp = torch.classes.recis.SparseAdamWTF.make(
            param_dict,
            self._lr,
            self._beta1,
            self._beta2,
            self._eps,
            self._weight_decay,
            self._use_nesterov,
        )
