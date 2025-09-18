import math
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
    _get_value,
    _use_grad_for_differentiable,
)


def maybe_get_value(inp):
    if isinstance(inp, torch.Tensor):
        return inp.item()
    else:
        return inp


class AdamWTF(Optimizer):
    """AdamW optimizer with TensorFlow-style implementation for dense parameters.

    This class implements the AdamW optimization algorithm with TensorFlow-compatible
    behavior for dense parameter optimization. It extends PyTorch's Optimizer base
    class and provides efficient optimization for standard neural network parameters.

    The AdamW algorithm combines adaptive learning rates from Adam with proper
    weight decay regularization. Unlike the original Adam optimizer, AdamW applies
    weight decay directly to the parameters rather than adding it to the gradients,
    which provides better regularization behavior especially for transformer models.

    Mathematical formulation:
    .. math::
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        m̂_t = m_t / (1 - β₁^t)
        v̂_t = v_t / (1 - β₂^t)
        θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + weight_decay * θ_{t-1})

    Where:
        - θ: parameters
        - g: gradients
        - m: first moment estimate (momentum)
        - v: second moment estimate (variance)
        - β₁, β₂: exponential decay rates
        - lr: learning rate
        - ε: numerical stability constant

    Key features:
    - TensorFlow-compatible behavior and numerical precision
    - Proper weight decay implementation (decoupled from gradients)
    - Optional Nesterov momentum support
    - Fused kernel optimization for better performance
    - Support for both scalar and tensor learning rates

    Example:
        Basic usage for transformer training:

    .. code-block:: python

        # Initialize for transformer model
        optimizer = AdamWTF(
            model.parameters(),
            lr=0.001,  # Learning rate
            betas=(0.9, 0.999),  # Adam momentum parameters
            eps=1e-8,  # Numerical stability
            weight_decay=0.01,  # L2 regularization strength
            use_nesterov=False,  # Standard Adam behavior
        )

        # Training loop
        for batch in dataloader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()

        Advanced configuration for different model types:
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        use_nesterov: bool = False,
        *,
        maximize: bool = False,
        fuse: bool = True,
    ):
        """Initialize AdamWTF optimizer with specified hyperparameters.

        Args:
            params (ParamsT): Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr (Union[float, Tensor], optional): Learning rate. Can be a scalar
                or tensor for dynamic learning rates. Defaults to 1e-3.
            betas (Tuple[float, float], optional): Coefficients used for computing
                running averages of gradient and its square. First value is beta1
                (momentum), second is beta2 (variance). Defaults to (0.9, 0.999).
            eps (float, optional): Term added to the denominator to improve
                numerical stability. Defaults to 1e-8.
            weight_decay (float, optional): Weight decay coefficient (L2 penalty).
                Applied directly to parameters, not gradients. Defaults to 1e-2.
            use_nesterov (bool, optional): Whether to use Nesterov momentum.
                Provides faster convergence in some cases. Defaults to False.
            maximize (bool, optional): Maximize the objective with respect to the
                params, instead of minimizing. Defaults to False.
            fuse (bool, optional): Whether to use fused kernel implementation
                for better performance. Defaults to True.

        Raises:
            ValueError: If any hyperparameter is outside valid range.

        Note:
            The fused implementation provides significant speedup on CUDA devices
            but may have slightly different numerical behavior compared to the
            non-fused version.
        """
        # Validate hyperparameters
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_nesterov=use_nesterov,
            maximize=maximize,
            fuse=fuse,
            differentiable=False,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("use_nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("fuse", True)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]), dtype=torch.float32)
        self.defaults["differentiable"] = False

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        use_nesterov,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            assert not torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AdamWTF does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])
        return False

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        This method executes one iteration of the AdamW optimization algorithm,
        updating all parameters based on their gradients and the current
        optimizer state.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss. Used for algorithms that require multiple
                function evaluations per step.

        Returns:
            Optional[float]: The loss value if closure is provided, None otherwise.

        Note:
            This method automatically handles parameter grouping, state
            initialization, and delegates to the appropriate implementation
            (fused or non-fused) based on the optimizer configuration.
        """
        # Check for CUDA graph capture compatibility
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            use_nesterov = group["use_nesterov"]
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                use_nesterov,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
            )

            adamwtf(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                use_nesterov=use_nesterov,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                fuse=group["fuse"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


AdamWTF.__doc__ = r"""Implements AdamWTF algorithm.
    """


def adamwtf(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    use_nesterov: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    fuse: bool,
):
    r"""Functional API that performs AdamWTF algorithm computation.

    See :class:`~torch.optim.AdamWTF` for details.
    """
    if not torch._utils.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )
    if fuse:
        func = _fuse_tensor_adamwtf
    else:
        func = _single_tensor_adamwtf
    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        use_nesterov=use_nesterov,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


def _fuse_tensor_adamwtf(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    use_nesterov: bool,
    beta1: float,
    beta2: float,
    lr: Union[Tensor, float],
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    assert grad_scale is None and found_inf is None
    assert not use_nesterov

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        assert not torch.is_complex(param)
        step_t += 1
        step = _get_value(step_t)
        lr_scalar = maybe_get_value(lr)
        # Perform stepweight decay
        param.mul_(1 - weight_decay)

        # Decay the first and second moment running average coefficient
        torch.ops.recis.adam_tf_apply(
            param, grad, exp_avg, exp_avg_sq, step, lr_scalar, beta1, beta2, eps
        )


def _single_tensor_adamwtf(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    use_nesterov: bool,
    beta1: float,
    beta2: float,
    lr: Union[Tensor, float],
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    assert grad_scale is None and found_inf is None
    assert not use_nesterov

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        assert not torch.is_complex(param)
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step = _get_value(step_t)
        lr_scalar = maybe_get_value(lr)

        b1_power = beta1**step
        b2_power = beta2**step

        bias_correction1 = 1 - b1_power
        bias_correction2 = 1 - b2_power

        alpha = lr_scalar / bias_correction1 * math.sqrt(bias_correction2)

        denom = exp_avg_sq.sqrt().add_(eps)

        param.addcdiv_(exp_avg, denom, value=-alpha)
