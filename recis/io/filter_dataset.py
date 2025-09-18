class _FilterDatasetIterator:
    """Iterator for FilterDataset that applies filtering functions to data.

    This internal iterator class handles the actual application of filter functions
    to data items as they are requested. It maintains the state of the underlying
    iterator and skips items that match any of the filter conditions.

    Attributes:
        _dataset (FilterDataset): Reference to the parent FilterDataset.
        _input_iterator (Iterator): The underlying data iterator.
        _filter_funcs (List[callable]): List of filter functions to apply.

    Note:
        This is an internal class used by FilterDataset and should not be
        instantiated directly by users.
    """

    def __init__(self, dataset, input_iterator, filter_funcs=None) -> None:
        """Initialize the filter dataset iterator.

        Args:
            dataset (FilterDataset): Reference to the parent FilterDataset instance.
            input_iterator (Iterator): The underlying iterator to filter.
            filter_funcs (List[callable], optional): List of filter functions to apply.
                Each function should return True for items to filter out. Defaults to None.

        Note:
            Filter functions are applied with logical OR - if any function returns
            True, the item is filtered out.
        """
        self._dataset = dataset
        self._input_iterator = input_iterator
        self._filter_funcs = filter_funcs if filter_funcs else []

    def __iter__(self):
        """Return the iterator object itself.

        Returns:
            _FilterDatasetIterator: Self reference for iterator protocol.
        """
        return self

    def __next__(self):
        """Get the next data item that passes all filter conditions.

        This method retrieves items from the underlying iterator and applies
        all filter functions. Items that match any filter condition are skipped,
        and the method continues until it finds an item that passes all filters.

        Returns:
            Any: The next data item that passes all filter conditions.

        Raises:
            StopIteration: When the underlying iterator is exhausted.

        Example:
            The filtering logic works as follows:
            ```
            for each item:
                if any(filter_func(item) for filter_func in filter_funcs):
                    skip item  # Filter out
                else:
                    return item  # Keep
            ```

        Note:
            This method may need to process multiple items from the underlying
            iterator before finding one that passes all filter conditions.
        """
        do_filter = False
        while True:
            ret = next(self._input_iterator)
            for filter_func in self._filter_funcs:
                do_filter |= filter_func(ret)
                if do_filter:
                    do_filter = False
                    continue
            break  # Skip this item and try next
        return ret


class FilterDataset:
    """Dataset wrapper that applies filtering functions to exclude unwanted data.

    FilterDataset provides a functional approach to data filtering by allowing
    users to specify a list of filter functions that determine which data items
    should be excluded from the stream. This enables flexible and composable
    data cleaning and preprocessing pipelines.

    The dataset uses lazy evaluation, meaning filtering is only applied when
    data is actually requested, which is memory efficient for large datasets.
    Filter functions are combined with logical OR - if any function returns True,
    the item is filtered out.

    Attributes:
        _input_dataset (Iterable): The source dataset to filter.
        _filter_funcs (List[callable]): List of filter functions.

    Example:
        Creating a data cleaning pipeline:

        ```python
        # Define data quality filters
        def filter_corrupted_data(item):
            # Filter out items with corrupted features
            return item.get("features") is None or len(item["features"]) == 0


        def filter_outliers(item):
            # Filter out statistical outliers
            value = item.get("value", 0)
            return abs(value) > 3 * std_threshold


        def filter_invalid_timestamps(item):
            # Filter out items with invalid timestamps
            timestamp = item.get("timestamp", 0)
            return timestamp < min_valid_time or timestamp > max_valid_time


        # Create comprehensive filtering pipeline
        filters = [filter_corrupted_data, filter_outliers, filter_invalid_timestamps]
        clean_dataset = FilterDataset(raw_dataset, filters)

        # Process only clean, valid data
        valid_count = 0
        for clean_item in clean_dataset:
            process_item(clean_item)
            valid_count += 1

        print(f"Processed {valid_count} valid items")
        ```

        Domain-specific filtering:

        ```python
        # Text processing filters
        def filter_short_text(item):
            return len(item.get("text", "").split()) < 5


        def filter_non_english(item):
            return not is_english(item.get("text", ""))


        def filter_spam(item):
            return is_spam_content(item.get("text", ""))


        # Create text filtering pipeline
        text_filters = [filter_short_text, filter_non_english, filter_spam]
        clean_text_dataset = FilterDataset(text_dataset, text_filters)

        # Only high-quality text samples will be processed
        for text_sample in clean_text_dataset:
            train_model(text_sample)
        ```

    Note:
        Filter functions should return True for items that should be excluded
        and False for items that should be kept. The filtering is applied
        lazily during iteration.
    """

    def __init__(self, input_dataset, filter_funcs=None) -> None:
        """Initialize FilterDataset with input dataset and filter functions.

        Args:
            input_dataset (Iterable): The source dataset to apply filters to.
                This can be any iterable object including other datasets.
            filter_funcs (List[callable], optional): List of filter functions to apply.
                Each function should accept a single data item and return True
                for items to filter out (exclude). Defaults to None.

        Example:
            ```python
            def remove_negatives(item):
                return item.get("value", 0) < 0


            def remove_duplicates(item):
                return item.get("id") in seen_ids


            dataset = FilterDataset(
                source_dataset, [remove_negatives, remove_duplicates]
            )
            ```

        Note:
            If filter_funcs is None or empty, the dataset will pass through all
            data unchanged. This allows for conditional filtering pipelines.
            Filter functions are applied with logical OR combination.
        """
        self._input_dataset = input_dataset
        self._filter_funcs = filter_funcs if filter_funcs else []

    def __iter__(self):
        """Create and return an iterator for the filtered dataset.

        Returns:
            _FilterDatasetIterator: An iterator that applies filter functions to
                exclude unwanted items from the input dataset.

        Note:
            Each call to __iter__ creates a new iterator instance, allowing
            multiple concurrent iterations over the same filtered dataset.
        """
        return _FilterDatasetIterator(
            self, iter(self._input_dataset), self._filter_funcs
        )
