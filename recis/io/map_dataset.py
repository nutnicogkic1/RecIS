class _MapDatasetIterator:
    """Iterator for MapDataset that applies transformation functions to data.

    This internal iterator class handles the actual application of map functions
    to data items as they are requested. It maintains the state of the underlying
    iterator and applies transformations in sequence.

    Attributes:
        _dataset (MapDataset): Reference to the parent MapDataset.
        _input_iterator (Iterator): The underlying data iterator.
        _map_funcs (List[callable]): List of transformation functions to apply.

    Note:
        This is an internal class used by MapDataset and should not be
        instantiated directly by users.
    """

    def __init__(self, dataset, input_iterator, map_funcs) -> None:
        """Initialize the map dataset iterator.

        Args:
            dataset (MapDataset): Reference to the parent MapDataset instance.
            input_iterator (Iterator): The underlying iterator to transform.
            map_funcs (List[callable]): List of transformation functions to apply
                in sequence to each data item.

        Note:
            The map functions are applied in the order they appear in the list,
            with each function receiving the output of the previous function.
        """
        self._dataset = dataset
        self._input_iterator = input_iterator
        self._map_funcs = map_funcs if map_funcs else []

    def __iter__(self):
        """Return the iterator object itself.

        Returns:
            _MapDatasetIterator: Self reference for iterator protocol.
        """
        return self

    def __next__(self):
        """Get the next transformed data item.

        This method retrieves the next item from the underlying iterator and
        applies all transformation functions in sequence.

        Returns:
            Any: The transformed data item after applying all map functions.

        Raises:
            StopIteration: When the underlying iterator is exhausted.

        Example:
            The transformation pipeline works as follows:
            ```
            original_item -> func1 -> func2 -> ... -> funcN -> final_item
            ```
        """
        ret = next(self._input_iterator)
        for map_func in self._map_funcs:
            ret = map_func(ret)
        return ret


class MapDataset:
    """Dataset wrapper that applies transformation functions to input data.

    MapDataset provides a functional approach to data transformation by allowing
    users to specify a list of transformation functions that are applied to each
    data item in sequence. This enables flexible and composable data preprocessing
    pipelines.

    The dataset uses lazy evaluation, meaning transformations are only applied
    when data is actually requested, which is memory efficient for large datasets.

    Attributes:
        _input_dataset (Iterable): The source dataset to transform.
        _map_funcs (List[callable]): List of transformation functions.

    Example:
        Creating a transformation pipeline:

        ```python
        # Define preprocessing functions
        def preprocess_text(item):
            item["text"] = item["text"].lower().strip()
            return item


        def tokenize(item):
            item["tokens"] = item["text"].split()
            return item


        def add_length(item):
            item["length"] = len(item["tokens"])
            return item


        # Create transformation pipeline
        transforms = [preprocess_text, tokenize, add_length]
        dataset = MapDataset(raw_dataset, transforms)

        # Process data with transformations applied
        for processed_item in dataset:
            print(f"Processed item with {processed_item['length']} tokens")
        ```

    Note:
        Each transformation function should accept a single data item and return
        the transformed item. Functions are applied in the order specified in
        the map_funcs list.
    """

    def __init__(self, input_dataset, map_funcs) -> None:
        """Initialize MapDataset with input dataset and transformation functions.

        Args:
            input_dataset (Iterable): The source dataset to apply transformations to.
                This can be any iterable object including other datasets.
            map_funcs (List[callable]): List of transformation functions to apply.
                Each function should accept a single data item and return the
                transformed item. Functions are applied in sequence.

        Example:
            ```python
            def normalize(item):
                item["value"] = (item["value"] - mean) / std
                return item


            def clip(item):
                item["value"] = max(-1.0, min(1.0, item["value"]))
                return item


            dataset = MapDataset(source_dataset, [normalize, clip])
            ```

        Note:
            If map_funcs is None or empty, the dataset will pass through data
            unchanged. This allows for conditional transformation pipelines.
        """
        self._input_dataset = input_dataset
        self._map_funcs = map_funcs if map_funcs else []

    def __iter__(self):
        """Create and return an iterator for the transformed dataset.

        Returns:
            _MapDatasetIterator: An iterator that applies transformations to
                each item from the input dataset.

        Note:
            Each call to __iter__ creates a new iterator instance, allowing
            multiple concurrent iterations over the same transformed dataset.
        """
        return _MapDatasetIterator(self, iter(self._input_dataset), self._map_funcs)
