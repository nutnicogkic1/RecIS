import json
from typing import Optional

import torch


class Initializer:
    """Base class for all parameter initializers.

    This abstract base class defines the common interface and functionality
    for all parameter initializers in RecIS. It provides methods for setting
    tensor shape and dtype, building the internal implementation, and
    generating initialized tensors.

    The class uses a builder pattern where the initializer is first configured
    with parameters, then built to create an internal implementation, and
    finally used to generate tensors with the specified initialization.
    """

    @classmethod
    def _avoid_rebuild_wrapper(cls, fn):
        """Decorator to prevent rebuilding an already built initializer.

        Args:
            fn: The function to wrap.

        Returns:
            function: Wrapped function that checks for rebuild attempts.
        """

        def wrapper(self, *args, **kwargs):
            if self._impl is not None:
                raise RuntimeError("rebuild initializer")
            return fn(self, *args, **kwargs)

        return wrapper

    def __init__(self) -> None:
        """Initialize the base initializer.

        Sets up the internal state with None values for implementation,
        shape, and dtype that will be configured later.
        """
        self._impl = None
        self._shape = None
        self._dtype = None

    def set_shape(self, shape: Optional[list] = None):
        """Set the shape of tensors to be generated.

        Args:
            shape (list, optional): List of integers specifying tensor dimensions.
                For example, [100, 50] for a 100x50 matrix.
        """
        self._shape = shape

    def set_dtype(self, dtype: torch.dtype = torch.float32):
        """Set the data type of tensors to be generated.

        Args:
            dtype (torch.dtype, optional): PyTorch data type for the tensors.
                Defaults to torch.float32.
        """
        self._dtype = dtype

    def build(self):
        """Build the internal implementation of the initializer.

        This method must be implemented by subclasses to create the
        internal generator implementation based on the configured
        parameters, shape, and dtype.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("build method is not implemented")

    def impl(self):
        """Get the internal implementation object.

        Returns:
            The internal generator implementation created by build().
        """
        return self._impl

    def generate(self) -> torch.Tensor:
        """Generate an initialized tensor.

        Creates and returns a tensor with the configured shape, dtype,
        and initialization values based on the initializer's strategy.

        Returns:
            torch.Tensor: Initialized tensor with the specified shape and dtype.

        Raises:
            RuntimeError: If build() has not been called before generate().
        """
        if self._impl is None:
            raise RuntimeError("build method is not called.")
        return torch.classes.recis.Generator.generate(self.impl())

    def __str__(self):
        """Return string representation of the initializer.

        This method must be implemented by subclasses to provide
        a JSON string representation of the initializer configuration.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("to_str method is not implemented.")


class ConstantInitializer(Initializer):
    """Constant value initializer.

    Initializes all parameters with the same constant value. This is useful
    for bias initialization or when you want all parameters to start with
    the same value.

    Args:
        init_val (float, optional): The constant value to initialize parameters.
            Defaults to 0.0.
        dtype (float, torch.dtype): The value type to initialize parameters.
            Defaults to torch.float32.

    Example:
        >>> # Initialize all parameters to 0.1
        >>> initializer = ConstantInitializer(init_val=0.1)
        >>> initializer.set_shape([10, 5])
        >>> initializer.build()
        >>> tensor = initializer.generate()  # All values will be 0.1
    """

    def __init__(
        self, init_val: float = 0.0, dtype: torch.dtype = torch.float32
    ) -> None:
        """Initialize the constant initializer.

        Args:
            init_val (float, optional): The constant value for initialization.
                Defaults to 0.0.
            dtype (torch.dtype, optional): The value type for initialization.
                Defaults to torch.float32.
        """
        super().__init__()
        self._init_val = init_val

    @Initializer._avoid_rebuild_wrapper
    def build(self):
        """Build the constant generator implementation.

        Creates an internal generator that produces tensors filled with
        the specified constant value.
        """
        self._impl = torch.classes.recis.Generator.make_constant_generator(
            self._shape, self._dtype, self._init_val
        )

    def __str__(self):
        """Return JSON string representation of the initializer.

        Returns:
            str: JSON string containing initializer configuration.
        """
        info = {
            "type": "ConstantInitializer",
            "init_val": self._init_val,
            "shape": str(self._shape),
            "dtype": str(self._dtype),
        }
        return json.dumps(info, sort_keys=True)


class UniformInitializer(Initializer):
    """Uniform distribution initializer.

    Initializes parameters by sampling from a uniform distribution over
    the interval [a, b). This provides a simple way to initialize parameters
    with values spread evenly across a specified range.

    Args:
        a (float, optional): Lower bound of the uniform distribution (inclusive).
            Defaults to 0.0.
        b (float, optional): Upper bound of the uniform distribution (exclusive).
            Defaults to 1.0.
        generator (torch.Generator, optional): Random number generator for
            reproducible initialization. Defaults to None.

    Example:
        >>> # Initialize parameters uniformly between -0.1 and 0.1
        >>> initializer = UniformInitializer(a=-0.1, b=0.1)
        >>> initializer.set_shape([100, 50])
        >>> initializer.build()
        >>> tensor = initializer.generate()
    """

    def __init__(
        self, a: float = 0.0, b: float = 1.0, generator: torch.Generator = None
    ) -> None:
        """Initialize the uniform distribution initializer.

        Args:
            a (float, optional): Lower bound (inclusive). Defaults to 0.0.
            b (float, optional): Upper bound (exclusive). Defaults to 1.0.
            generator (torch.Generator, optional): Random generator for
                reproducible results. Defaults to None.
        """
        super().__init__()
        self._a = a
        self._b = b
        self._generator = generator

    @Initializer._avoid_rebuild_wrapper
    def build(self):
        """Build the uniform distribution generator implementation.

        Creates an internal generator that produces tensors with values
        sampled from the specified uniform distribution.
        """
        self._impl = torch.classes.recis.Generator.make_uniform_generator(
            self._shape, self._dtype, self._a, self._b, self._generator
        )

    def __str__(self):
        """Return JSON string representation of the initializer.

        Returns:
            str: JSON string containing initializer configuration.
        """
        info = {
            "type": "UniformInitializer",
            "a": self._a,
            "b": self._b,
            "shape": str(self._shape),
            "dtype": str(self._dtype),
            "generator_seed": self._generator.seed() if self._generator else None,
        }
        return json.dumps(info, sort_keys=True)


class NormalInitializer(Initializer):
    """Normal (Gaussian) distribution initializer.

    Initializes parameters by sampling from a normal distribution with
    specified mean and standard deviation. This is one of the most common
    initialization strategies for neural networks.

    Args:
        mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
        std (float, optional): Standard deviation of the normal distribution.
            Defaults to 1.0.
        generator (torch.Generator, optional): Random number generator for
            reproducible initialization. Defaults to None.

    Example:
        >>> # Initialize with small random values around zero
        >>> initializer = NormalInitializer(mean=0.0, std=0.01)
        >>> initializer.set_shape([784, 128])
        >>> initializer.build()
        >>> tensor = initializer.generate()
    """

    def __init__(
        self, mean: float = 0.0, std: float = 1.0, generator: torch.Generator = None
    ) -> None:
        """Initialize the normal distribution initializer.

        Args:
            mean (float, optional): Mean of the distribution. Defaults to 0.0.
            std (float, optional): Standard deviation. Defaults to 1.0.
            generator (torch.Generator, optional): Random generator for
                reproducible results. Defaults to None.
        """
        super().__init__()
        self._mean = mean
        self._std = std
        self._generator = generator

    @Initializer._avoid_rebuild_wrapper
    def build(self):
        """Build the normal distribution generator implementation.

        Creates an internal generator that produces tensors with values
        sampled from the specified normal distribution.
        """
        self._impl = torch.classes.recis.Generator.make_normal_generator(
            self._shape, self._dtype, self._mean, self._std, self._generator
        )

    def __str__(self):
        """Return JSON string representation of the initializer.

        Returns:
            str: JSON string containing initializer configuration.
        """
        info = {
            "type": "NormalInitializer",
            "mean": self._mean,
            "std": self._std,
            "shape": str(self._shape),
            "dtype": str(self._dtype),
            "generator_seed": self._generator.seed() if self._generator else None,
        }
        return json.dumps(info, sort_keys=True)


class XavierUniformInitializer(Initializer):
    """Xavier/Glorot uniform distribution initializer.

    Initializes parameters using Xavier initialization with uniform distribution.
    This method is designed to keep the scale of gradients roughly the same
    in all layers by sampling from a uniform distribution with bounds calculated
    based on the number of input and output units.

    The bounds are calculated as: ±sqrt(6 / (fan_in + fan_out)) * gain

    Args:
        gain (float, optional): Scaling factor for the initialization range.
            Defaults to 1.0.
        generator (torch.Generator, optional): Random number generator for
            reproducible initialization. Defaults to None.

    Example:
        >>> # Xavier initialization for a linear layer
        >>> initializer = XavierUniformInitializer(gain=1.0)
        >>> initializer.set_shape([256, 128])  # 256 inputs, 128 outputs
        >>> initializer.build()
        >>> tensor = initializer.generate()
    """

    def __init__(self, gain: float = 1.0, generator: torch.Generator = None) -> None:
        """Initialize the Xavier uniform initializer.

        Args:
            gain (float, optional): Scaling factor. Defaults to 1.0.
            generator (torch.Generator, optional): Random generator for
                reproducible results. Defaults to None.
        """
        super().__init__()
        self._gain = gain
        self._generator = generator

    @Initializer._avoid_rebuild_wrapper
    def build(self):
        """Build the Xavier uniform generator implementation.

        Creates an internal generator that produces tensors initialized
        according to Xavier uniform initialization strategy.
        """
        self._impl = torch.classes.recis.Generator.make_xavier_uniform_generator(
            self._shape, self._dtype, self._gain, self._generator
        )

    def __str__(self):
        """Return JSON string representation of the initializer.

        Returns:
            str: JSON string containing initializer configuration.
        """
        info = {
            "type": "XavierUniformInitializer",
            "gain": self._gain,
            "shape": str(self._shape),
            "dtype": str(self._dtype),
            "generator_seed": self._generator.seed() if self._generator else None,
        }
        return json.dumps(info, sort_keys=True)


class XavierNormalInitializer(Initializer):
    """Xavier/Glorot normal distribution initializer.

    Initializes parameters using Xavier initialization with normal distribution.
    Similar to XavierUniformInitializer but uses a normal distribution instead
    of uniform. The standard deviation is calculated as: sqrt(2 / (fan_in + fan_out)) * gain

    Args:
        gain (float, optional): Scaling factor for the initialization range.
            Defaults to 1.0.
        generator (torch.Generator, optional): Random number generator for
            reproducible initialization. Defaults to None.

    Example:
        >>> # Xavier normal initialization with custom gain
        >>> initializer = XavierNormalInitializer(gain=1.414)  # sqrt(2)
        >>> initializer.set_shape([512, 256])
        >>> initializer.build()
        >>> tensor = initializer.generate()
    """

    def __init__(self, gain: float = 1.0, generator: torch.Generator = None) -> None:
        """Initialize the Xavier normal initializer.

        Args:
            gain (float, optional): Scaling factor. Defaults to 1.0.
            generator (torch.Generator, optional): Random generator for
                reproducible results. Defaults to None.
        """
        super().__init__()
        self._gain = gain
        self._generator = generator

    @Initializer._avoid_rebuild_wrapper
    def build(self):
        """Build the Xavier normal generator implementation.

        Creates an internal generator that produces tensors initialized
        according to Xavier normal initialization strategy.
        """
        self._impl = torch.classes.recis.Generator.make_xavier_normal_generator(
            self._shape, self._dtype, self._gain, self._generator
        )

    def __str__(self):
        """Return JSON string representation of the initializer.

        Returns:
            str: JSON string containing initializer configuration.
        """
        info = {
            "type": "XavierNormalInitializer",
            "gain": self._gain,
            "shape": str(self._shape),
            "dtype": str(self._dtype),
            "generator_seed": self._generator.seed() if self._generator else None,
        }
        return json.dumps(info, sort_keys=True)


class KaimingUniformInitializer(Initializer):
    """Kaiming uniform distribution initializer (He initialization).

    Initializes parameters using Kaiming/He initialization with uniform distribution.
    This method is particularly effective for layers with ReLU activations as it
    accounts for the variance reduction caused by ReLU's zero-clipping behavior.

    The bounds are calculated based on the fan-in or fan-out and the nonlinearity
    function used in the network.

    Args:
        a (float, optional): Negative slope of the rectifier used after this layer
            (only used with 'leaky_relu'). Defaults to 0.0.
        mode (str, optional): Either 'fan_in' or 'fan_out'. Choosing 'fan_in'
            preserves the magnitude of the variance of the weights in the forward
            pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
            Defaults to 'fan_in'.
        nonlinearity (str, optional): The non-linear function (nn.functional name),
            recommended to use only with 'relu' or 'leaky_relu'. Defaults to 'leaky_relu'.
        generator (torch.Generator, optional): Random number generator for
            reproducible initialization. Defaults to None.

    Example:
        >>> # Kaiming initialization for ReLU activation
        >>> initializer = KaimingUniformInitializer(
        ...     a=0.0, mode="fan_in", nonlinearity="relu"
        ... )
        >>> initializer.set_shape([1024, 512])
        >>> initializer.build()
        >>> tensor = initializer.generate()
    """

    def __init__(
        self,
        a: float = 0.0,
        mode: str = "fan_in",
        nonlinearity="leaky_relu",
        generator: torch.Generator = None,
    ) -> None:
        """Initialize the Kaiming uniform initializer.

        Args:
            a (float, optional): Negative slope parameter. Defaults to 0.0.
            mode (str, optional): 'fan_in' or 'fan_out'. Defaults to 'fan_in'.
            nonlinearity (str, optional): Nonlinearity function name.
                Defaults to 'leaky_relu'.
            generator (torch.Generator, optional): Random generator for
                reproducible results. Defaults to None.
        """
        super().__init__()
        self._a = a
        self._mode = mode
        self._nonlinearity = nonlinearity
        self._generator = generator

    @Initializer._avoid_rebuild_wrapper
    def build(self):
        """Build the Kaiming uniform generator implementation.

        Creates an internal generator that produces tensors initialized
        according to Kaiming uniform initialization strategy.
        """
        self._impl = torch.classes.recis.Generator.make_kaiming_uniform_generator(
            self._shape,
            self._dtype,
            self._a,
            self._mode.lower(),
            self._nonlinearity.lower(),
            self._generator,
        )

    def __str__(self):
        """Return JSON string representation of the initializer.

        Returns:
            str: JSON string containing initializer configuration.
        """
        info = {
            "type": "KaimingUniformInitializer",
            "a": self._a,
            "mode": self._mode,
            "nonlinearity": self._nonlinearity,
            "shape": str(self._shape),
            "dtype": str(self._dtype),
            "generator_seed": self._generator.seed() if self._generator else None,
        }
        return json.dumps(info, sort_keys=True)


class KaimingNormalInitializer(Initializer):
    """Kaiming normal distribution initializer (He initialization).

    Initializes parameters using Kaiming/He initialization with normal distribution.
    Similar to KaimingUniformInitializer but uses normal distribution instead of
    uniform. This is often preferred for deep networks with ReLU activations.

    Args:
        shape (list, optional): Shape of the tensor to initialize. This parameter
            appears to be unused in the current implementation. Defaults to None.
        dtype (torch.dtype, optional): Data type of the tensor. This parameter
            appears to be unused in the current implementation. Defaults to torch.float32.
        a (float, optional): Negative slope of the rectifier used after this layer
            (only used with 'leaky_relu'). Defaults to 0.0.
        mode (str, optional): Either 'fan_in' or 'fan_out'. Defaults to 'fan_in'.
        nonlinearity (str, optional): The non-linear function name.
            Defaults to 'leaky_relu'.
        generator (torch.Generator, optional): Random number generator for
            reproducible initialization. Defaults to None.

    Example:
        >>> # Kaiming normal initialization for deep ReLU network
        >>> initializer = KaimingNormalInitializer(
        ...     a=0.0, mode="fan_out", nonlinearity="relu"
        ... )
        >>> initializer.set_shape([2048, 1024])
        >>> initializer.build()
        >>> tensor = initializer.generate()
    """

    def __init__(
        self,
        shape: Optional[list] = None,
        dtype: torch.dtype = torch.float32,
        a: float = 0.0,
        mode: str = "fan_in",
        nonlinearity="leaky_relu",
        generator: torch.Generator = None,
    ) -> None:
        """Initialize the Kaiming normal initializer.

        Args:
            shape (list, optional): Tensor shape (unused). Defaults to None.
            dtype (torch.dtype, optional): Data type (unused). Defaults to torch.float32.
            a (float, optional): Negative slope parameter. Defaults to 0.0.
            mode (str, optional): 'fan_in' or 'fan_out'. Defaults to 'fan_in'.
            nonlinearity (str, optional): Nonlinearity function name.
                Defaults to 'leaky_relu'.
            generator (torch.Generator, optional): Random generator for
                reproducible results. Defaults to None.
        """
        super().__init__()
        self._a = a
        self._mode = mode
        self._nonlinearity = nonlinearity
        self._generator = generator

    @Initializer._avoid_rebuild_wrapper
    def build(self):
        """Build the Kaiming normal generator implementation.

        Note: The current implementation uses make_kaiming_uniform_generator
        which appears to be a bug. This should likely use a normal distribution
        generator instead.
        """
        self._impl = torch.classes.recis.Generator.make_kaiming_uniform_generator(
            self._shape,
            self._dtype,
            self._a,
            self._mode.lower(),
            self._nonlinearity.lower(),
            self._generator,
        )

    def __str__(self):
        """Return JSON string representation of the initializer.

        Returns:
            str: JSON string containing initializer configuration.
        """
        info = {
            "type": "KaimingNormalInitializer",
            "a": self._a,
            "mode": self._mode,
            "nonlinearity": self._nonlinearity,
            "shape": str(self._shape),
            "dtype": str(self._dtype),
            "generator_seed": self._generator.seed() if self._generator else None,
        }
        return json.dumps(info, sort_keys=True)


class TruncNormalInitializer(Initializer):
    """Truncated normal distribution initializer.

    Initializes parameters by sampling from a truncated normal distribution.
    This is similar to normal initialization but with values outside the
    specified range [a, b] resampled to ensure all values fall within bounds.
    This can be useful when you want normally distributed values but need
    to avoid extreme outliers.

    Args:
        mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
        std (float, optional): Standard deviation of the normal distribution.
            Defaults to 1.0.
        a (float, optional): Lower truncation bound in units of standard deviations
            from the mean. Defaults to -2.0.
        b (float, optional): Upper truncation bound in units of standard deviations
            from the mean. Defaults to 2.0.
        generator (torch.Generator, optional): Random number generator for
            reproducible initialization. Defaults to None.

    Example:
        >>> # Truncated normal with small std and tight bounds
        >>> initializer = TruncNormalInitializer(mean=0.0, std=0.02, a=-2.0, b=2.0)
        >>> initializer.set_shape([512, 256])
        >>> initializer.build()
        >>> tensor = initializer.generate()  # Values in [-0.04, 0.04]
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        a: float = -2.0,
        b: float = 2.0,
        generator: torch.Generator = None,
    ) -> None:
        """Initialize the truncated normal initializer.

        Args:
            mean (float, optional): Mean of the distribution. Defaults to 0.0.
            std (float, optional): Standard deviation. Defaults to 1.0.
            a (float, optional): Lower bound in std units. Defaults to -2.0.
            b (float, optional): Upper bound in std units. Defaults to 2.0.
            generator (torch.Generator, optional): Random generator for
                reproducible results. Defaults to None.
        """
        super().__init__()
        self._mean = mean
        self._std = std
        self._a = a
        self._b = b
        self._generator = generator

    @Initializer._avoid_rebuild_wrapper
    def build(self):
        """Build the truncated normal generator implementation.

        Creates an internal generator that produces tensors with values
        sampled from the specified truncated normal distribution.
        """
        self._impl = torch.classes.recis.Generator.make_trunc_normal_generator(
            self._shape,
            self._dtype,
            self._mean,
            self._std,
            self._a,
            self._b,
            self._generator,
        )

    def __str__(self):
        """Return JSON string representation of the initializer.

        Returns:
            str: JSON string containing initializer configuration.
        """
        info = {
            "type": "TruncNormalInitializer",
            "mean": self._mean,
            "std": self._std,
            "a": self._a,
            "b": self._b,
            "shape": str(self._shape),
            "dtype": str(self._dtype),
            "generator_seed": self._generator.seed() if self._generator else None,
        }
        return json.dumps(info, sort_keys=True)
