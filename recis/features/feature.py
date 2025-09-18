import hashlib
from typing import Dict, List

from torch import nn

from .op import _OP, SelectField, SelectFields


class Feature(nn.Module):
    """A feature processing pipeline that encapsulates a sequence of operations.

    The Feature class represents a single feature in a machine learning pipeline,
    containing a sequence of operations that transform input data. Features can
    be compiled for optimization and provide hash-based caching for efficiency.

    For example:

    .. code-block:: python

        from recis.features import FeatureEngine
        from recis.features.feature import Feature
        from recis.features.op import SelectField, Hash, Bucketize

        # Define features
        features = [
            Feature("user_id").add_op(SelectField("user_id")).add_op(Mod(10000)),
            Feature("age")
            .add_op(SelectField("age"))
            .add_op(Bucketize(boundaries=[18, 25, 35, 45, 55])),
        ]

        # Create feature engine
        feature_engine = FeatureEngine(features)

        # Data processing
        input_data = {
            "user_id": torch.LongTensor([1, 2, 3]),
            "age": torch.FloatTensor([20, 30, 40]),
        }

        output_data = feature_engine(input_data)

    """

    def __init__(self, name: str):
        """Initialize a new feature with the given name.

        Args:
            name (str): The unique identifier name for this feature.
        """
        super().__init__()
        self._name = name
        self._compiled = False
        self._ops = nn.ModuleList()
        self._buffers_data = None
        self._input_flag = False

    def set_buffers_data(self, data):
        self._buffers_data = data

    @property
    def id(self):
        return id(self)

    @property
    def name(self):
        return self._name

    @property
    def compiled(self):
        return self._compiled

    def compiled_(self, value: bool):
        assert isinstance(value, bool)
        self._compiled = value

    @property
    def ops(self) -> List[nn.Module]:
        return self._ops

    def add_op(self, op: _OP):
        """Add an operation to this feature's processing pipeline.

        Operations are executed in the order they are added. The first operation
        must be a SelectField or SelectFields operation. Dependencies of the
        added operation are automatically included.

        Args:
            op (_OP): The operation to add to the pipeline.

        Returns:
            Feature: This feature instance for method chaining.

        Raises:
            ValueError: If the feature has already been compiled or if the first
                       operation is not a SelectField/SelectFields operation.
        """
        if self._compiled:
            raise ValueError(f"feature {self.name} has been compiled")
        if len(self._ops) == 0:
            if type(op) in [SelectField, SelectFields]:
                self._input_flag = True
            else:
                raise ValueError(f"feature {self.name} must start with DataInputOP")
        self._ops.append(op)

        return self

    @staticmethod
    def from_json(json_obj: Dict):
        raise NotImplementedError

    def get_hash(self) -> int:
        if len(self._ops) == 0:
            return 0

        op_hashes = []
        for op in self._ops:
            op_hashes.append(op.get_hash())

        combined_str = f"{self._name}:{sorted(op_hashes)}"
        hash_bytes = hashlib.sha256(combined_str.encode("utf-8")).digest()
        hash_value = int.from_bytes(hash_bytes[:8], byteorder="big", signed=True)

        return hash_value

    def forward(self, data):
        """Execute the feature processing pipeline on input data.

        Applies all operations in the pipeline sequentially to transform
        the input data according to the feature definition.

        Args:
            data: Input data to be processed. The format depends on the
                 first operation in the pipeline.

        Returns:
            The transformed data after applying all operations in sequence.
        """
        x = data
        for op in self._ops:
            x = op(x)
        return x
