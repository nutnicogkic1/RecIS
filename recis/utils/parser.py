from simple_parsing import ArgumentParser as SAP


class ArgumentParser(SAP):
    """Enhanced argument parser with automatic destination handling.

    This class extends simple-parsing's ArgumentParser to provide automatic
    destination parameter setting. When no explicit destination is provided,
    it automatically uses the class name as the destination, making the
    argument parsing more intuitive and consistent.

    The parser inherits all functionality from simple-parsing's ArgumentParser
    while adding convenience features for RecIS framework usage.

    Features:
        - Automatic destination parameter setting based on class name
        - Full compatibility with simple-parsing ArgumentParser
        - Support for dataclass-based configuration objects
        - Type-safe argument parsing with automatic validation

    Example:
        >>> from recis.utils.parser import ArgumentParser
        >>> from dataclasses import dataclass
        >>> @dataclass
        >>> class ModelConfig:
        ...     hidden_size: int = 128
        ...     num_layers: int = 3
        ...     dropout: float = 0.1
        >>> @dataclass
        >>> class OptimizerConfig:
        ...     learning_rate: float = 0.001
        ...     weight_decay: float = 0.01
        >>> # Create parser and add configurations
        >>> parser = ArgumentParser()
        >>> parser.add_arguments(ModelConfig)  # dest will be "ModelConfig"
        >>> parser.add_arguments(OptimizerConfig)  # dest will be "OptimizerConfig"
        >>> # Parse arguments
        >>> args = parser.parse_args(
        ...     [
        ...         "--ModelConfig.hidden_size",
        ...         "256",
        ...         "--OptimizerConfig.learning_rate",
        ...         "0.01",
        ...     ]
        ... )
        >>> # Access parsed configurations
        >>> print(f"Hidden size: {args.ModelConfig.hidden_size}")
        >>> print(f"Learning rate: {args.OptimizerConfig.learning_rate}")
    """

    def add_arguments(self, *args, **kwargs):
        """Add arguments to the parser with automatic destination handling.

        This method extends the parent's add_arguments method by automatically
        setting the destination parameter when it's not explicitly provided.
        The destination is set to the class name of the first positional argument.

        Args:
            *args: Positional arguments passed to the parent method. The first
                argument should be a class (typically a dataclass) that defines
                the configuration structure.
            **kwargs: Keyword arguments passed to the parent method. If 'dest'
                is not provided, it will be automatically set to the name of
                the first positional argument's class.

        Returns:
            The result of the parent's add_arguments method.

        Example:
            >>> @dataclass
            >>> class Config:
            ...     param1: int = 10
            ...     param2: str = "default"
            >>> parser = ArgumentParser()
            >>> # Automatic destination (will use "Config")
            >>> parser.add_arguments(Config)
            >>> # Manual destination
            >>> parser.add_arguments(Config, dest="custom_config")
            >>> # Parse and access
            >>> args = parser.parse_args()
            >>> print(args.Config.param1)  # Using automatic dest
            >>> print(args.custom_config.param1)  # Using manual dest
        """
        if kwargs.get("dest", None) is None:
            kwargs["dest"] = args[0].__name__
        return super().add_arguments(*args, **kwargs)
