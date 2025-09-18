import torch

from recis.optim.sparse_optim import SparseOptimizer


class SparseAdamW(SparseOptimizer):
    """Sparse AdamW optimizer for efficient sparse parameter optimization.

    This class implements the AdamW optimization algorithm specifically optimized
    for sparse parameters in recommendation systems. It extends the SparseOptimizer
    base class and uses RecIS's C++ implementation for maximum performance.

    The AdamW algorithm combines adaptive learning rates from Adam with proper
    weight decay regularization. For sparse parameters, this implementation only
    updates parameters that have received gradients, making it highly efficient
    for large embedding tables where only a small fraction of parameters are
    active in each training step.

    Mathematical formulation:

    .. math::

        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t \n
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t² \n
        m̂_t = m_t / (1 - β₁^t) \n
        v̂_t = v_t / (1 - β₂^t) \n
        θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + weight_decay * θ_{t-1})

    Where:
        - θ: parameters
        - g: gradients
        - m: first moment estimate (momentum)
        - v: second moment estimate (variance)
        - β₁, β₂: exponential decay rates
        - lr: learning rate
        - ε: numerical stability constant


    Example:
        Creating and using SparseAdamW:

    .. code-block:: python

        # Initialize with custom hyperparameters
        optimizer = SparseAdamW(
            param_dict=sparse_parameters,
            lr=0.001,  # Learning rate
            beta1=0.9,  # First moment decay rate
            beta2=0.999,  # Second moment decay rate
            eps=1e-8,  # Numerical stability
            weight_decay=0.01,  # L2 regularization strength
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
        """Initialize SparseAdamW optimizer with specified hyperparameters.

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
            weight_decay (float, optional): Weight decay coefficient (L2 regularization).
                Defaults to 1e-2.
            amsgrad (bool, optional): Whether to use AMSGrad variant. Currently not
                supported and will raise ValueError if True. Defaults to False.

        Raises:
            ValueError: If amsgrad is True (not currently supported).

        Example:

        .. code-block:: python

            # Basic initialization
            optimizer = SparseAdamW(sparse_params)

            # Custom hyperparameters for recommendation systems
            optimizer = SparseAdamW(
                param_dict=embedding_params,
                lr=0.01,  # Higher learning rate for sparse params
                beta1=0.9,  # Standard momentum
                beta2=0.999,  # Standard variance decay
                eps=1e-8,  # Numerical stability
                weight_decay=0.001,  # Light regularization
            )

            # Conservative settings for fine-tuning
            optimizer = SparseAdamW(
                param_dict=pretrained_embeddings,
                lr=0.0001,  # Low learning rate
                weight_decay=0.1,  # Strong regularization
            )


        Note:
            The param_dict should contain HashTable parameters that support
            sparse gradient updates. Regular dense tensors may not work correctly
            with this optimizer.
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
        self._imp = torch.classes.recis.SparseAdamW.make(
            param_dict,
            self._lr,
            self._beta1,
            self._beta2,
            self._eps,
            self._weight_decay,
            self._amsgrad,
        )
