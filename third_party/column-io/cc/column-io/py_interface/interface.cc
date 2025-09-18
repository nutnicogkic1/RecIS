#include "absl/log/initialize.h"
#include "column-io/dataset/dataset.h"
#include "column-io/dataset/list_dataset.h"
#include "column-io/dataset/list_combo_dataset.h"
#include "column-io/dataset/packer.h"
#include "column-io/dataset/parallel_dataset.h"
#include "column-io/dataset/prefetch_dataset.h"
#include "column-io/dataset/repeat_dataset.h"
#include "column-io/dataset_impl/local_rb_stream_dataset.h"
#include "column-io/dataset_impl/local_orc_dataset.h"
#if INTERNAL_VERSION
#include "column-io/dataset_impl/lake_stream_column_dataset.h"
#include "column-io/dataset_impl/lake_batch_column_dataset.h"
#if (_GLIBCXX_USE_CXX11_ABI == 0)
#include "column-io/dataset_impl/odps_table_column_dataset.h"
#else
#include "column-io/dataset_impl/odps_open_storage_dataset.h"
#endif
#include "column-io/dataset_impl/odps_table_column_combo_dataset.h"
#include "column-io/lake/lake_fslib_helper.h"
#include "column-io/odps/proxy/lib_odps.h"
//#include "column-io/open_storage/wrapper/dl_wrapper_open_storage.h"
//#include "column-io/open_storage/wrapper/odps_open_storage_arrow_reader.h"
#if (_GLIBCXX_USE_CXX11_ABI != 0)
#include "column-io/py_interface/open_storage_wrapper.h"
#endif
#endif
#include "column-io/framework/types.h"
#include "column-io/py_interface/converter.h"
#include "column-io/py_interface/dataset.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <memory>
#include <pybind11/detail/common.h>
namespace py = pybind11;
PYBIND11_MODULE(py_interface, m) {
#if INTERNAL_VERSION
  #if (_GLIBCXX_USE_CXX11_ABI != 0)
  //open storage
  m.def("GetOdpsOpenStorageTableSize", column::open_storage::GetOdpsOpenStorageTableSize);
  m.def("InitOdpsOpenStorageSessions", column::open_storage::InitOdpsOpenStorageSessions);
  m.def("RegisterOdpsOpenStorageSession", column::open_storage::RegisterOdpsOpenStorageSession);
  m.def("ExtractLocalReadSession", column::open_storage::ExtractLocalReadSession);
  m.def("RefreshReadSessionBatch", column::open_storage::RefreshReadSessionBatch);
  m.def("GetOdpsOpenStorageTableFeatures", column::open_storage::GetOdpsOpenStorageTableFeatures);
  m.def("GetSessionExpireTimestamp", column::dataset::OdpsOpenStorageDataset::GetSessionExpireTimestamp);
  m.def("FreeBuffer", column::open_storage::FreeBuffer);
  #endif
#endif

  m.def("_global_init", column::dataset::GlobalInit);
  // iterator
  py::class_<column::dataset::IteratorBase,
             std::shared_ptr<column::dataset::IteratorBase>>(m, "_Iterator");
  m.def("MakeIterator", column::dataset::MakeIterator);
  m.def("GetNextFromIterator", &column::dataset::GetNextFromIterator,
        pybind11::return_value_policy::move);
  m.def("SerializeIteraterStateToString",
        &column::dataset::SerializeIteraterStateToString);
  m.def("DerializeIteraterStateFromString",
        &column::dataset::DeserializeIteratorStateFromString);

  // dataset
  py::class_<column::dataset::DatasetBase,
             std::shared_ptr<column::dataset::DatasetBase>>(m, "_Dataset");
  py::class_<column::dataset::DatasetBuilder,
             std::shared_ptr<column::dataset::DatasetBuilder>>(
      m, "_DatasetBuilder");

  // internal dataset.
  py::class_<column::dataset::ListStringDataset>(m, "_ListStringDataset")
      .def_static("make_dataset",
                  column::dataset::ListStringDataset::MakeDataset);

  py::class_<column::dataset::Packer>(m, "_PackerDataset")
      .def_static("make_dataset", column::dataset::Packer::MakeDataset)
      .def_static("make_reorder_dataset",
                  column::dataset::Packer::MakeReorderDataset);

  py::class_<column::dataset::ParallelDataset>(m, "_ParallelDataset")
      .def_static("make_dataset",
                  column::dataset::ParallelDataset::MakeDataset);

  py::class_<column::dataset::RepeatDataset>(m, "_RepeatDataset")
      .def_static("make_dataset", column::dataset::RepeatDataset::MakeDataset);

  py::class_<column::dataset::PrefetchDataset>(m, "_PrefetchDataset")
      .def_static("make_dataset",
                  column::dataset::PrefetchDataset::MakeDataset);

  py::class_<column::dataset::LocalRBStreamDataset>(m, "_LocalRBStreamDataset")
      .def_static("make_dataset",
                  column::dataset::LocalRBStreamDataset::MakeDatasetWrapper)
      .def_static("make_builder",
                  column::dataset::LocalRBStreamDataset::MakeBuilder)
      .def_static("parse_schema",
                  column::dataset::LocalRBStreamDataset::ParseSchema);

#if INTERNAL_VERSION
  // source dataset.
  #if (_GLIBCXX_USE_CXX11_ABI == 0)
  py::class_<column::dataset::OdpsTableColumnDataset>(m,
                                                      "_OdpsTableColumnDataset")
      .def_static("make_dataset",
                  column::dataset::OdpsTableColumnDataset::MakeDatasetWrapper)
      .def_static("make_builder",
                  column::dataset::OdpsTableColumnDataset::MakeBuilder)
      .def_static("parse_schema",
                  column::dataset::OdpsTableColumnDataset::ParseSchema)
      .def_static("get_table_size", column::dataset::GetTableSize)
      .def_static("load_odps_plugin", column::odps::proxy::LibOdps::LoadWrap);
  #endif

  #if (_GLIBCXX_USE_CXX11_ABI != 0)
  // odps-openstorage dataset
  py::class_<column::dataset::OdpsOpenStorageDataset>(m,
                                                      "_OdpsOpenStorageDataset")

      .def_static("make_dataset",
                  column::dataset::OdpsOpenStorageDataset::MakeDatasetWrapper)
      .def_static("make_builder",
                  column::dataset::OdpsOpenStorageDataset::MakeBuilder)
      .def_static("parse_schema",
                  column::dataset::OdpsOpenStorageDataset::ParseSchema)
      .def_static("get_table_size", column::dataset::GetTableSize)
      .def_static("load_open_storage_plugin", apsara::odps::tunnel::algo::tf::OdpsOpenStorageLib::LoadWrap);
  #endif

  #if (_GLIBCXX_USE_CXX11_ABI == 0)
  // combo dataset
  py::class_<column::dataset::ListStringComboDataset>(m, "_ListStringComboDataset")
      .def_static("make_dataset",
                  column::dataset::ListStringComboDataset::MakeDataset);

  // combo dataset
  py::class_<column::dataset::OdpsTableColumnComboDataset>(m,
                                                          "_OdpsTableColumnComboDataset")
      .def_static("make_dataset",
                  column::dataset::OdpsTableColumnComboDataset::MakeDatasetWrapper)
      .def_static("make_builder",
                  column::dataset::OdpsTableColumnComboDataset::MakeBuilder)
      .def_static("parse_schema",
                  column::dataset::OdpsTableColumnComboDataset::ParseSchema)
      .def_static("get_table_size", column::dataset::GetTableSize)
      .def_static("load_odps_plugin", column::odps::proxy::LibOdps::LoadWrap)
      .def_static("get_table_features", column::dataset::OdpsTableColumnDataset::GetOdpsTableFeatures);
  #endif

  py::class_<column::dataset::LakeStreamColumnDatase>(
      m, "_LakeStreamColumnDataset")
      .def_static("make_dataset",
                  column::dataset::LakeStreamColumnDatase::MakeDatasetWrapper)
      .def_static("make_builder",
                  column::dataset::LakeStreamColumnDatase::MakeBuilder)
      .def_static("parse_schema",
                  column::dataset::LakeStreamColumnDatase::ParseSchema)
      .def_static("close_pangu", lake::closePangu);

  py::class_<column::dataset::LakeBatchColumnDatase>(
      m, "_LakeBatchColumnDataset")
      .def_static("make_dataset",
                  column::dataset::LakeBatchColumnDatase::MakeDatasetWrapper)
      .def_static("make_builder",
                  column::dataset::LakeBatchColumnDatase::MakeBuilder)
      .def_static("parse_schema",
                  column::dataset::LakeBatchColumnDatase::ParseSchema)
      .def_static("parse_schema_by_rows",
                  column::dataset::LakeBatchColumnDatase::ParseSchemaByRows)
      .def_static("close_pangu", lake::closePangu);

#else
  py::class_<column::dataset::LocalOrcDataset>(m, "_LocalOrcDataset")
      .def_static("make_dataset",
                  column::dataset::LocalOrcDataset::MakeDatasetWrapper)
      .def_static("make_builder",
                  column::dataset::LocalOrcDataset::MakeBuilder)
      .def_static("parse_schema",
                  column::dataset::LocalOrcDataset::ParseSchema);
#endif
}
