import os
from typing import Any, Callable
import multiprocessing
import json

from column_io.lib import interface
from column_io.dataset.nest import pack_nest_sequence, _pack_nest_sequence_internal
from column_io.dataset.nest import nest_seq_leaf_num
try:
    from column_io.dataset.odps_env_setup import ensure_standard_path_format, \
        is_turn_on_odps_open_storage, init_odps_open_storage_session
    from column_io.dataset.log_util import logger, init_openstorage_logger
except:
    pass

kPlaceHolder = None


class Dataset:
    def __init__(self, impl) -> None:
        self._impl = impl

    def __iter__(self):
        iterator = interface.MakeIterator(self.impl())
        return Iterator(iterator, self)

    def impl(self):
        """
        get c++ reference of Dataset,
          user should not access this.
        """
        return self._impl

    @property
    def schema(self):
        raise NotImplemented("out_names not implemented")

    @staticmethod
    def from_list_string(array):
        """
        Args:
          array: a list of filenames.
        """
        return SliceListStringDataset(array)
    
    @staticmethod
    def from_list_string_combo(array):
        """
        Args:
          array: a list of odps_table_group.
        """
        return SliceListStringComboDataset(array)

    @staticmethod
    def from_rb_files(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        return LocalRBStreamDataset(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @staticmethod
    def from_orc_files(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        return LocalOrcDataset(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )


    @staticmethod
    def from_odps_source(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_xrec=False,
    ):
        """
        Args:
          paths: a list of filenames.
          is_compressed: specify if the data source is compressed.
          batch_size: the max batch size expected to read from odps.
          selected_columns: specity all the columns need to read.
          hash_features: specify the feature need to do hash
            for fast copy.
          hash_types: hash functions such as: farm, murmur.
          hash_buckets: buckets to hash.
          dense_columns: specify the dense columns.
          dense_defaults: specify the default value for dense columns.
          use_xrec: flag suggesting the user is xrec, print slice
            opening and out of range info.
        """
        if batch_size <= 0:
            raise ValueError("batch size must > 0 but get [{}]".format(batch_size))
        if is_turn_on_odps_open_storage():
            odps_dataset_func = OdpsOpenStorageDataset
        else:
            odps_dataset_func = OdpsTableColumnDataset
        return odps_dataset_func(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            use_xrec,
        )
    
    @staticmethod
    def from_open_storage_source(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ):
        """
        Args:
          paths: a list of filenames.
          is_compressed: specify if the data source is compressed.
          batch_size: the max batch size expected to read from odps.
          selected_columns: specity all the columns need to read.
          hash_features: specify the feature need to do hash
            for fast copy.
          hash_types: hash functions such as: farm, murmur.
          hash_buckets: buckets to hash.
          dense_columns: specify the dense columns.
          dense_defaults: specify the default value for dense columns.
        """
        if batch_size <= 0:
            raise ValueError("batch size must > 0 but get [{}]".format(batch_size))
        return OdpsOpenStorageDataset(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )
    
    @staticmethod
    def from_common_io_odps_source(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        dense_columns,
        dense_defaults,
    ):
        """
        Args:
          paths: a list of filenames.
          is_compressed: specify if the data source is compressed.
          batch_size: the max batch size expected to read from odps.
          selected_columns: specity all the columns need to read.
          hash_features: specify the feature need to do hash
            for fast copy.
          dense_columns: specify the dense columns.
          dense_defaults: specify the default value for dense columns.
        """
        if batch_size <= 0:
            raise ValueError("batch size must > 0 but get [{}]".format(batch_size))
        return Dataset.from_odps_source(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            hash_features,
            [], [],
            dense_columns,
            dense_defaults,
        )

    @staticmethod
    def from_odps_combo_source(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        check_data,
        primary_key
    ):
        """
        Args:
          paths: a list of filenames.
          is_compressed: specify if the data source is compressed.
          batch_size: the max batch size expected to read from odps.
          selected_columns: specity all the columns need to read.
          hash_features: specify the feature need to do hash
            for fast copy.
          hash_types: hash functions such as: farm, murmur.
          hash_buckets: buckets to hash.
          dense_columns: specify the dense columns.
          dense_defaults: specify the default value for dense columns.
        """
        if batch_size <= 0:
            raise ValueError("batch size must > 0 but get [{}]".format(batch_size))
        return OdpsTableColumnComboDataset(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            check_data,
            primary_key,
        )

    @staticmethod
    def from_lake_source(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_prefetch=False,
        prefetch_thread_num=1,
        prefetch_buffer_size=1024,
    ):
        """
        Args:
          paths: a list of filenames.
          is_compressed: specify if the data source is compressed.
          batch_size: the max batch size expected to read from odps.
          selected_columns: specity all the columns need to read.
          hash_features: specify the feature need to do hash
            for fast copy.
          hash_types: hash functions such as: farm, murmur.
          hash_buckets: buckets to hash.
          dense_columns: specify the dense columns.
          dense_defaults: specify the default value for dense columns.
          use_prefetch: use lake prefetch or not.
          prefetch_thread_num: lake prefetch thread num.
          prefetch_buffer_size: lake prefetch buffer size.
        """
        if batch_size <= 0:
            raise ValueError("batch size must > 0 but get [{}]".format(batch_size))
        return LakeStreamColumnDataset(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            use_prefetch,
            prefetch_thread_num,
            prefetch_buffer_size,
        )

    @staticmethod
    def from_lake_batch_source(
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_prefetch=False,
        prefetch_thread_num=1,
        prefetch_buffer_size=1024,
    ):
        """
        Args:
          paths: a list of filenames.
          is_compressed: specify if the data source is compressed.
          batch_size: the max batch size expected to read from odps.
          selected_columns: specity all the columns need to read.
          hash_features: specify the feature need to do hash
            for fast copy.
          hash_types: hash functions such as: farm, murmur.
          hash_buckets: buckets to hash.
          dense_columns: specify the dense columns.
          dense_defaults: specify the default value for dense columns.
          use_prefetch: use lake prefetch or not.
          prefetch_thread_num: lake prefetch thread num.
          prefetch_buffer_size: lake prefetch buffer size.
        """
        if batch_size <= 0:
            raise ValueError("batch size must > 0 but get [{}]".format(batch_size))
        return LakeBatchColumnDataset(
            paths,
            is_compressed,
            batch_size,
            selected_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            use_prefetch,
            prefetch_thread_num,
            prefetch_buffer_size,
        )
    
    def parallel(
        self,
        transfunc,
        cycle_length,
        block_length=1,
        sloppy=True,
        buffer_output_elements=1,
        prefetch_input_elements=0,
    ):
        """
        Args:
          transfunc: a sample map function accept a path as arg
            and return a dataset like lambda x: Dataset.from_odps_source([x],....),
            now only support Dataset.from_odps_source.
          cycle_length: The number of input `Dataset`s to interleave from in parallel.
          block_length: The number of consecutive elements to pull from an input
            `Dataset` before advancing to the next input `Dataset`.
          sloppy: If false, elements are produced in deterministic order. Otherwise,
            the implementation is allowed, for the sake of expediency, to produce
            elements in a non-deterministic order.
          buffer_output_elements: The number of elements each iterator being
            interleaved should buffer (similar to the `.prefetch()` transformation for
            each interleaved iterator).
          prefetch_input_elements: The number of input elements to transform to
            iterators before they are needed for interleaving.
        """
        return ParallelDataset(
            self,
            transfunc,
            cycle_length,
            block_length,
            sloppy,
            buffer_output_elements,
            prefetch_input_elements,
        )

    def pack(self, batch_size, drop_remainder, parallel=None, pinned_result=False, gpu_result=False):
        """
        pack will pack the output of dataset to
          specified `batch_size`.
        Args:
          batch_size: the needed batch size.
          drop_remainder: specify if the reset of data should be dropped
            when the reset of data is not enough to build a output with `batch_size`
          parallel: size of thread pool to process, if None if will be set to
            number of cpu cores
          pinned_result: If true, the packed result will be on pinned memory.
          gpu_result: If true, the packed result will be on gpu memory.
            This argument will overwrite `pinned_result`.
        """
        if parallel is None:
            parallel = multiprocessing.cpu_count()
        if parallel <= 0:
            raise ValueError("parallel must > 0 but get [{}]".format(parallel))
        if batch_size <= 0:
            raise ValueError("batch size must > 0 but get [{}]".format(batch_size))
        return PackDataset(self, batch_size, drop_remainder, parallel, pinned_result, gpu_result)

    def repeat(self, take_num=1, repeat=-1):
        """
        take `take_num` batch from source dataset,
          and repeat `repeat` times.
        Args:
          take_num: the number of batch to take
            from source dataset.
          repeat: the repeat times on cached
            dataset, `-1` means repeat infinitly.
        """
        return RepeatDataset(self, take_num, repeat)

    def prefetch(self, buffer_size=1):
        """
        prefetch `buffer_size` batch from source dataset
          to make full use of cpu.
        Args:
          buffer_size: number of batch to take from
            source dataset.
        """
        return PrefetchDataset(self, buffer_size)


class Iterator:
    def __init__(self, iterator_impl, dataset: Dataset) -> None:
        self._iterator_impl = iterator_impl
        self._iterator_row_mode :bool = os.environ.get("ODPS_DATASET_ROW_MODE", "0") == "1" # TODO: support arg-style configuration
        self._dataset = dataset

    def __next__(self):
        # combo_mode
        if isinstance(self._dataset, SliceListStringComboDataset):
            return _pack_nest_sequence_internal(self.schema, interface.GetNextFromIterator(self._iterator_impl, self._iterator_row_mode), lambda x: x, 0)[0]

        # row_mode needn't pack array output according to schema, just keep list of row format
        # col_mode,however. need pack array output according to schema, reorder into map dict from name to col-batch
        if self._iterator_row_mode:
            # type: list[tuple[object]]
            return interface.GetNextFromIterator(self._iterator_impl, self._iterator_row_mode)
        else:
            # type: map[string, array[object]]
            return pack_nest_sequence(
                self.schema, interface.GetNextFromIterator(self._iterator_impl, self._iterator_row_mode)
            )

    @property
    def schema(self):
        return self._dataset.schema

    def serialize(self):
        """
        serialize the states of iterator to string.
        Retruns:
          a string
        NOTE: protobuf is used, the size of state should not be too large
          or a empty string will returned.
        """
        return interface.SerializeIteraterStateToString(self._iterator_impl)

    def deserialize(self, states):
        """
        deserialize the states of iterator from string
        Args:
          states: a string contain the state of iterator.
        """
        interface.DerializeIteraterStateFromString(self._iterator_impl, states)


class SliceListStringDataset(Dataset):
    _internal = interface._ListStringDataset

    def __init__(self, array) -> None:
        super().__init__(self._internal.make_dataset(array))

    def schema(self):
        return kPlaceHolder

class SliceListStringComboDataset(Dataset):
    _internal = interface._ListStringComboDataset

    def __init__(self, array) -> None:
        super().__init__(self._internal.make_dataset(array))

    def schema(self):
        return kPlaceHolder

class ParallelDataset(Dataset):
    _internal = interface._ParallelDataset

    def __init__(
        self,
        input: Dataset,
        transfunc,
        cycle_length,
        block_length,
        sloppy,
        buffer_output_elements,
        prefetch_input_elements,
    ) -> None:
        self._input = input
        out_dataset = transfunc(next(iter(input)))
        self._input_builder = out_dataset.builder
        if not issubclass(type(out_dataset), Dataset):
            raise RuntimeError(
                "output type of transfunc must be of type {}".format(type(Dataset))
            )
        self._schema = out_dataset.schema
        super().__init__(
            self._internal.make_dataset(
                input.impl(),
                self._input_builder,
                cycle_length,
                block_length,
                sloppy,
                buffer_output_elements,
                prefetch_input_elements,
            )
        )

    @property
    def schema(self):
        return self._schema


class PackDataset(Dataset):
    _internal = interface._PackerDataset

    def __init__(
        self, input: Dataset, batch_size, drop_remainder=False, parallel=None, pinned_result=False, gpu_result=False
    ) -> None:
        self._input = input
        self._make_reorder_info()
        self._preorder_dataset = self._internal.make_reorder_dataset(
            self._input.impl(), self._new_indice
        )
        super().__init__(
            self._internal.make_dataset(
                self._preorder_dataset,
                batch_size,
                drop_remainder,
                self._pack_tables,
                self._num_tables,
                self._ragged_ranks,
                parallel,
                pinned_result,
                gpu_result
            )
        )
        self._postorder_dataset = self._internal.make_reorder_dataset(
            self._impl, self._reverse_indice
        )

    def _make_reorder_info(self):
        input_schema = self._input.schema
        elem_num = nest_seq_leaf_num(input_schema)
        elem_indice = list(range(elem_num))
        schema_map_with_pos = pack_nest_sequence(input_schema, elem_indice)
        self._pack_tables = []
        self._num_tables = len(schema_map_with_pos)
        self._values_t = []
        self._splits_t = []
        self._names_t = []
        self._ragged_ranks = []
        self._indicators_t = []
        for table_idx, dic in enumerate(schema_map_with_pos):
            for feature in sorted(dic):
                positions_tuple = dic[feature]
                for positions in positions_tuple:
                    if feature.startswith("_indicator"):
                        self._indicators_t.append(positions[0])
                        continue
                    self._names_t.append(feature)
                    self._values_t.extend(positions[:1])
                    self._splits_t.extend(positions[1:])
                    self._ragged_ranks.append(len(positions) - 1)
                    self._pack_tables.append(table_idx)
        self._new_indice = self._indicators_t + self._values_t + self._splits_t
        indice_map = {new: ori for ori, new in enumerate(self._new_indice)}
        self._reverse_indice = [
            indice_map[index] for index in range(len(self._new_indice))
        ]

    def impl(self):
        return self._postorder_dataset

    @property
    def schema(self):
        return self._input.schema


class RepeatDataset(Dataset):
    _internal = interface._RepeatDataset

    def __init__(self, input: Dataset, take_num=1, repeat=-1):
        self._input = input
        super().__init__(self._internal.make_dataset(input.impl(), take_num, repeat))

    @property
    def schema(self):
        return self._input.schema


class PrefetchDataset(Dataset):
    _internal = interface._PrefetchDataset

    def __init__(self, input: Dataset, buffer_size=1) -> None:
        self._input = input
        super().__init__(self._internal.make_dataset(input.impl(), buffer_size))

    @property
    def schema(self):
        return self._input.schema


class OdpsTableColumnDataset(Dataset):
    _internal = interface._OdpsTableColumnDataset

    def __init__(
        self,
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_xrec,
    ) -> None:
        self._input_columns, self._schema = self._internal.parse_schema(
            paths,
            is_compressed,
            set(selected_columns),
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )
        super().__init__(
            self._internal.make_dataset(
                paths,
                is_compressed,
                batch_size,
                selected_columns,
                self._input_columns,
                hash_features,
                hash_types,
                hash_buckets,
                dense_columns,
                dense_defaults,
            )
        )

        self._builder = self._internal.make_builder(
            is_compressed,
            batch_size,
            selected_columns,
            self._input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @property
    def schema(self):
        return self._schema

    @property
    def builder(self):
        return self._builder

    @staticmethod
    def get_table_size(path):
        return OdpsTableColumnDataset._internal.get_table_size(path)
    
    @staticmethod
    def load_plugin():
      interface._OdpsTableColumnDataset.load_plugin()

class OdpsOpenStorageDataset(Dataset):
    _internal = interface._OdpsOpenStorageDataset

    def __init__(
        self,
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_xrec,
    ) -> None:
        init_openstorage_logger()
        standard_paths = ensure_standard_path_format(paths)
        init_odps_open_storage_session(standard_paths, required_data_columns=selected_columns)
        self._input_columns, self._schema = self._internal.parse_schema(
            paths,
            is_compressed,
            set(selected_columns),
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )
        super().__init__(
            self._internal.make_dataset(
                paths,
                is_compressed,
                batch_size,
                selected_columns,
                self._input_columns,
                hash_features,
                hash_types,
                hash_buckets,
                dense_columns,
                dense_defaults,
                use_xrec,
            )
        )

        self._builder = self._internal.make_builder(
            is_compressed,
            batch_size,
            selected_columns,
            self._input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            use_xrec,
        )

    @property
    def schema(self):
        return self._schema

    @property
    def builder(self):
        return self._builder

    @staticmethod
    def get_table_size(path):
        return interface._OdpsOpenStorageDataset.get_table_size(path)

    @staticmethod
    def get_session_expire_timestamp(session_id):
        # type: (str)->int
        return interface._OdpsOpenStorageDataset.get_session_expire_timestamp(session_id)

    @staticmethod
    def load_plugin():
      interface._OdpsOpenStorageDataset.load_plugin()


class OdpsTableColumnComboDataset(Dataset):
    _internal = interface._OdpsTableColumnComboDataset
    def __init__(
        self,
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        check_data,
        primary_key
    ) -> None:
        table_group = paths[0]
        table_columns = [[] for _ in range(len(table_group))]
        table_schemas = []
        output_schema = {}
        table_schemas = self._fetch_schema(table_group, is_compressed)
        
        for feature in selected_columns:
            table_index = -1
            for i in range(len(table_group)-1, -1, -1):
                if feature in table_schemas[i]:
                    table_index = i
                    break
            if table_index == -1:
                raise ValueError("can not find feature [{}]".format(feature))
            table_columns[table_index].append(feature)
        for i in range(0, len(table_group)) :
            input_columns, schema = self._internal.parse_schema(
                [table_group[i]],
                is_compressed,
                set(table_columns[i]),
                hash_features,
                hash_types,
                hash_buckets,
                dense_columns,
                dense_defaults
            )
            output_schema.update(schema[0])
        self._schema = [output_schema]
        print("self._schema : ", self._schema)
        
        super().__init__(
            self._internal.make_dataset(
                paths,
                is_compressed,
                batch_size,
                selected_columns,
                # self._input_columns,
                table_columns,
                hash_features,
                hash_types,
                hash_buckets,
                dense_columns,
                dense_defaults,
                check_data,
                primary_key,
            )
        )
        self._builder = self._internal.make_builder(
            is_compressed,
            batch_size,
            selected_columns,
            table_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
            check_data,
            primary_key,
        )
    def _fetch_schema(self, table_group, _is_compressed):
        """Get schemas of all tables from the provided table group"""
        table_schemas = []
        for tg in table_group:
            cur_schema = interface._OdpsTableColumnDataset.get_table_features(tg, _is_compressed)
            table_schemas.append(cur_schema)
        return table_schemas
    @property
    def schema(self):
        return self._schema
    @property
    def builder(self):
        return self._builder
    @staticmethod
    def get_table_size(path):
        return OdpsTableColumnDataset._internal.get_table_size(path)
    
    @staticmethod
    def load_plugin():
      interface._OdpsTableColumnDataset.load_plugin()

class LocalRBStreamDataset(Dataset):
    _internal = interface._LocalRBStreamDataset

    def __init__(
        self,
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ) -> None:
        self._input_columns, self._schema = self._internal.parse_schema(
            paths,
            is_compressed,
            set(selected_columns),
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

        super().__init__(
            self._internal.make_dataset(
                paths,
                is_compressed,
                batch_size,
                selected_columns,
                self._input_columns,
                hash_features,
                hash_types,
                hash_buckets,
                dense_columns,
                dense_defaults,
            )
        )

        self._builder = self._internal.make_builder(
            is_compressed,
            batch_size,
            selected_columns,
            self._input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @property
    def schema(self):
        return self._schema

    @property
    def builder(self):
        return self._builder


class LocalOrcDataset(Dataset):
    _internal = interface._LocalOrcDataset

    def __init__(
        self,
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
    ) -> None:
        self._input_columns, self._schema = self._internal.parse_schema(
            paths,
            is_compressed,
            set(selected_columns),
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

        super().__init__(
            self._internal.make_dataset(
                paths,
                is_compressed,
                batch_size,
                selected_columns,
                self._input_columns,
                hash_features,
                hash_types,
                hash_buckets,
                dense_columns,
                dense_defaults,
            )
        )

        self._builder = self._internal.make_builder(
            is_compressed,
            batch_size,
            selected_columns,
            self._input_columns,
            hash_features,
            hash_types,
            hash_buckets,
            dense_columns,
            dense_defaults,
        )

    @property
    def schema(self):
        return self._schema

    @property
    def builder(self):
        return self._builder


class LakeStreamColumnDataset(Dataset):
    _internal = interface._LakeStreamColumnDataset

    def __init__(
        self,
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_prefetch,
        prefetch_thread_num,
        prefetch_buffer_size,
    ):
        self._input_columns, self._schema = self._internal.parse_schema(
            paths=paths,
            is_compressed=is_compressed,
            selected_columns=set(selected_columns),
            hash_features=hash_features,
            hash_types=hash_types,
            hash_buckets=hash_buckets,
            dense_columns=dense_columns,
            dense_defaults=dense_defaults,
        )

        super().__init__(
            self._internal.make_dataset(
                paths=paths,
                is_compressed=is_compressed,
                batch_size=batch_size,
                selected_columns=selected_columns,
                input_columns=self._input_columns,
                hash_features=hash_features,
                hash_types=hash_types,
                hash_buckets=hash_buckets,
                dense_columns=dense_columns,
                dense_defaults=dense_defaults,
                use_prefetch=use_prefetch,
                prefetch_thread_num=prefetch_thread_num,
                prefetch_buffer_size=prefetch_buffer_size,
            )
        )

        self._builder = self._internal.make_builder(
            is_compressed=is_compressed,
            batch_size=batch_size,
            selected_columns=selected_columns,
            input_columns=self._input_columns,
            hash_features=hash_features,
            hash_types=hash_types,
            hash_buckets=hash_buckets,
            dense_columns=dense_columns,
            dense_defaults=dense_defaults,
            use_prefetch=use_prefetch,
            prefetch_thread_num=prefetch_thread_num,
            prefetch_buffer_size=prefetch_buffer_size,
        )

    @property
    def schema(self):
        return self._schema

    @property
    def builder(self):
        return self._builder


class LakeBatchColumnDataset(Dataset):
    _internal = interface._LakeBatchColumnDataset

    def __init__(
        self,
        paths,
        is_compressed,
        batch_size,
        selected_columns,
        hash_features,
        hash_types,
        hash_buckets,
        dense_columns,
        dense_defaults,
        use_prefetch,
        prefetch_thread_num,
        prefetch_buffer_size,
    ):
        iterator_row_mode :bool = os.environ.get("ODPS_DATASET_ROW_MODE", "0") == "1" # TODO: support arg-style configuration
        parse_schema_func : callable # type: Callable[[str], tuple[str, dict[str, str]] ]
        if not iterator_row_mode:
            parse_schema_func = self._internal.parse_schema
            schema_selected_columns=set(selected_columns)
        else:
            parse_schema_func = self._internal.parse_schema_by_rows
            schema_selected_columns=selected_columns
        
        self._input_columns, self._schema = parse_schema_func(
            paths=paths,
            is_compressed=is_compressed,
            selected_columns=schema_selected_columns,
            hash_features=hash_features,
            hash_types=hash_types,
            hash_buckets=hash_buckets,
            dense_columns=dense_columns,
            dense_defaults=dense_defaults,
        )

        # Allow empty to select all columns
        if len(selected_columns) == 0:
            selected_columns = self._input_columns

        super().__init__(
            self._internal.make_dataset(
                paths=paths,
                is_compressed=is_compressed,
                batch_size=batch_size,
                selected_columns=selected_columns,
                input_columns=self._input_columns,
                hash_features=hash_features,
                hash_types=hash_types,
                hash_buckets=hash_buckets,
                dense_columns=dense_columns,
                dense_defaults=dense_defaults,
                use_prefetch=use_prefetch,
                prefetch_thread_num=prefetch_thread_num,
                prefetch_buffer_size=prefetch_buffer_size,
            )
        )

        self._builder = self._internal.make_builder(
            is_compressed=is_compressed,
            batch_size=batch_size,
            selected_columns=selected_columns,
            input_columns=self._input_columns,
            hash_features=hash_features,
            hash_types=hash_types,
            hash_buckets=hash_buckets,
            dense_columns=dense_columns,
            dense_defaults=dense_defaults,
            use_prefetch=use_prefetch,
            prefetch_thread_num=prefetch_thread_num,
            prefetch_buffer_size=prefetch_buffer_size,
        )

    @property
    def schema(self):
        return self._schema

    @property
    def builder(self):
        return self._builder
