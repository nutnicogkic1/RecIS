import torch


class CheckpointReader:
    """Provides read access to checkpoint files and their metadata.

    This class serves as a wrapper around the low-level torch.classes.recis.CheckpointReader,
    offering a convenient interface to inspect and read tensors from checkpoint files.

    Examples:
        Typical usage example for reading checkpoint contents:

        >>> reader = CheckpointReader("/path/to/checkpoint")
        >>> tensor_names = reader.tensor_names()
        >>> for name in tensor_names:
        ...     shape = reader.tensor_shape(name)
        ...     dtype = reader.tensor_dtype(name)
        ...     tensor_data = reader.read_tensor(name)

    Attributes:
        reader: The underlying implementation object handling low-level checkpoint reading.
    """

    def __init__(self, path):
        """Initializes the CheckpointReader with a path to checkpoint files.

        Args:
            path: The directory path containing checkpoint files to read.

        Note:
            The reader initialization may involve loading metadata and preparing
            for subsequent read operations.
        """
        self.reader = torch.classes.recis.CheckpointReader(path)
        self.reader.init()

    def tensor_names(self):
        """Retrieves the names of all tensors available in the checkpoint.

        Returns:
            A list of string identifiers for all tensors stored in the checkpoint.
        """
        return self.reader.list_tensor_names()

    def read_tensor(self, name):
        """Reads and returns the tensor data for the specified tensor name.

        Args:
            name: The identifier of the tensor to read.

        Returns:
            The tensor data as an appropriate array or tensor object.

        Raises:
            KeyError: If the specified tensor name does not exist in the checkpoint.
        """
        return self.reader.read_tensor(name)

    def tensor_shape(self, name):
        """Retrieves the shape/dimensions of the specified tensor.

        Args:
            name: The identifier of the tensor.

        Returns:
            A tuple representing the shape of the tensor.

        Raises:
            KeyError: If the specified tensor name does not exist in the checkpoint.
        """
        return self.reader.tensor_shape(name)

    def tensor_dtype(self, name):
        """Retrieves the data type of the specified tensor.

        Args:
            name: The identifier of the tensor.

        Returns:
            The data type object representing the tensor's element type.

        Raises:
            KeyError: If the specified tensor name does not exist in the checkpoint.
        """
        return self.reader.tensor_dtype(name).dtype
