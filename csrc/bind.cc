#include <string>
#include <unordered_map>

#include "ATen/core/Dict.h"
#include "ATen/core/TensorBody.h"
#include "embedding/adam.h"
#include "embedding/adamw.h"
#include "embedding/adamw_tf.h"
#include "embedding/children_info.h"
#include "embedding/hashtable.h"
#include "embedding/initializer.h"
#include "embedding/optim.h"
#include "embedding/slot_group.h"
#include "ops/adam_tf_op.h"
#include "ops/block_apply_adamw_op.h"
#include "ops/block_ops.h"
#include "ops/bucketize_op.h"
#include "ops/calc_ragged_index.h"
#include "ops/dense_to_ragged.h"
#include "ops/embedding_segment_reduce.h"
#include "ops/feature_cross_ragged.h"
#include "ops/fused_bucketized.h"
#include "ops/fused_hash.h"
#include "ops/fused_ragged_cutoff.h"
#include "ops/fused_uint64_mod.h"
#include "ops/gauc.h"
#include "ops/hashtable_ops.h"
#include "ops/ids_encode.h"
#include "ops/ids_partition.h"
#include "ops/multi_hash.h"
#include "ops/ragged_tile.h"
#include "ops/ragged_to_dense.h"
#include "ops/ragged_to_sparse.h"
#include "ops/segment_ops.h"
#include "ops/uint64_mod.h"
#include "serialize/checkpoint_reader.h"
#include "serialize/loader.h"
#include "serialize/saver.h"
#include "serialize/write_block.h"
#include "torch/csrc/utils/pybind.h"
#include "torch/custom_class.h"
#include "torch/extension.h"
#include "torch/library.h"
#include "torch/types.h"

TORCH_LIBRARY(recis, m) {
  m.class_<recis::embedding::ChildrenInfo>("ChildrenInfo")
      .def("index_bits_num", &recis::embedding::ChildrenInfo::IndexBitsNum)
      .def("id_bits_num", &recis::embedding::ChildrenInfo::IdBitsNum)
      .def("index_mask", &recis::embedding::ChildrenInfo::IndexMask)
      .def("id_mask", &recis::embedding::ChildrenInfo::IdMask)
      .def("max_children_num", &recis::embedding::ChildrenInfo::MaxChildrenNum)
      .def("encode_id", &recis::embedding::ChildrenInfo::EncodeId)
      .def("children", &recis::embedding::ChildrenInfo::Children)
      .def("child_index", &recis::embedding::ChildrenInfo::ChildIndex)
      .def("has_child", &recis::embedding::ChildrenInfo::HasChild)
      .def("child_at", &recis::embedding::ChildrenInfo::ChildAt)
      .def("is_coalesce", &recis::embedding::ChildrenInfo::IsCoalesce);

  m.class_<recis::embedding::Slot>("Slot")
      .def("block_size", &recis::embedding::Slot::BlockSize)
      .def("name", &recis::embedding::Slot::Name)
      .def("flat_size", &recis::embedding::Slot::FlatSize)
      .def("value", &recis::embedding::Slot::Value)
      .def("values", &recis::embedding::Slot::ValuesDeref);

  m.class_<recis::embedding::SlotGroup>("SlotGroup")
      .def("slot_by_name", &::recis::embedding::SlotGroup::GetSlotByName)
      .def("slot_by_index", &::recis::embedding::SlotGroup::GetSlotByIndex)
      .def("slot_index", &::recis::embedding::SlotGroup::GetSlotByIndex)
      .def("slots", &::recis::embedding::SlotGroup::Slots);

  m.class_<recis::embedding::Generator>("Generator")
      .def_static("generate", recis::embedding::Generator::DoGenerator)
      .def_static("make_constant_generator",
                  recis::embedding::MakeConstantGenerator)
      .def_static("make_uniform_generator",
                  recis::embedding::MakeUniformGenerator)
      .def_static("make_normal_generator",
                  recis::embedding::MakeNormalGenerator)
      .def_static("make_xavier_uniform_generator",
                  recis::embedding::MakeXavierUniFormGenerator)
      .def_static("make_xavier_normal_generator",
                  recis::embedding::MakeXavierNormalGenerator)
      .def_static("make_kaiming_uniform_generator",
                  recis::embedding::MakeKaimingUniformGenerator)
      .def_static("make_kaiming_normal_generator",
                  recis::embedding::MakeKaimingNormalGenerator)
      .def_static("make_trunc_normal_generator",
                  recis::embedding::MakeTruncNormalGenerator);

  m.class_<Hashtable>("HashtableImpl")
      .def("accept_grad", &Hashtable::AcceptGrad)
      .def("grad", &Hashtable::grad)
      .def("clear_grad", &Hashtable::ClearGrad)
      .def("embedding_lookup", &Hashtable::EmbeddingLookup)
      .def("insert", &Hashtable::Insert)
      .def("clear", &Hashtable::Clear)
      .def("clear_id", &Hashtable::ClearId)
      .def("ids", &Hashtable::ids)
      .def("slot_group", &Hashtable::SlotGroup)
      .def("children_info", &Hashtable::ChildrenInfo)
      .def("append_filter_slot", &Hashtable::AppendFilterSlot)
      .def("append_step_filter_slot", &Hashtable::AppendStepFilterSlot)
      .def("update_slot", &Hashtable::UpdateSlot)
      .def("get_slot", &Hashtable::GetSlot)
      .def("delete", &Hashtable::Delete)
      .def("ids_map", &Hashtable::ids_map)
      .def_readonly("hashtable_tag", &Hashtable::kNullIndex);

  // optimzier
  m.class_<recis::optim::SparseAdamW>("SparseAdamW")
      .def("step", &recis::optim::SparseAdamW::step)
      .def("add_parameters", &recis::optim::SparseAdamW::add_parameters)
      .def("zero_grad", &recis::optim::SparseAdamW::zero_grad)
      .def("state_dict", &recis::optim::SparseAdamW::state_dict)
      .def("load_state_dict", &recis::optim::SparseAdamW::load_state_dict)
      .def("set_grad_accum_steps",
           &recis::optim::SparseAdamW::set_grad_accum_steps)
      .def("set_lr", &recis::optim::SparseAdamW::set_lr)
      .def_static("make", recis::optim::SparseAdamW::Make);

  m.class_<recis::optim::SparseAdam>("SparseAdam")
      .def("step", &recis::optim::SparseAdam::step)
      .def("add_parameters", &recis::optim::SparseAdam::add_parameters)
      .def("zero_grad", &recis::optim::SparseAdam::zero_grad)
      .def("state_dict", &recis::optim::SparseAdam::state_dict)
      .def("load_state_dict", &recis::optim::SparseAdam::load_state_dict)
      .def("set_grad_accum_steps",
           &recis::optim::SparseAdam::set_grad_accum_steps)
      .def("set_lr", &recis::optim::SparseAdam::set_lr)
      .def_static("make", recis::optim::SparseAdam::Make);

  m.class_<recis::optim::SparseAdamWTF>("SparseAdamWTF")
      .def("step", &recis::optim::SparseAdamWTF::step)
      .def("add_parameters", &recis::optim::SparseAdamWTF::add_parameters)
      .def("zero_grad", &recis::optim::SparseAdamWTF::zero_grad)
      .def("state_dict", &recis::optim::SparseAdamWTF::state_dict)
      .def("load_state_dict", &recis::optim::SparseAdamWTF::load_state_dict)
      .def("set_grad_accum_steps",
           &recis::optim::SparseAdamWTF::set_grad_accum_steps)
      .def("set_lr", &recis::optim::SparseAdamWTF::set_lr)
      .def_static("make", recis::optim::SparseAdamWTF::Make);

  m.def("make_hashtable", Hashtable::Make);

  m.class_<recis::serialize::WriteBlock>("WriteBlock")
      .def("tensor_name", &recis::serialize::WriteBlock::TensorName)
      .def("slice_info", &recis::serialize::WriteBlock::SliceInfo);

  m.class_<recis::serialize::Saver>("Saver")
      .def(torch::init<int64_t, int64_t, int64_t, const std::string>())
      .def("make_write_blocks", &recis::serialize::Saver::MakeWriteBlocks)
      .def("save", &recis::serialize::Saver::Save);

  m.class_<recis::serialize::Loader>("Loader")
      .def(torch::init<const std::string, int64_t,
                       torch::Dict<std::string, HashTablePtr>,
                       torch::Dict<std::string, torch::Tensor>>())
      .def("default_load_info", &recis::serialize::Loader::DefaultLoadInfo)
      .def("load", &recis::serialize::Loader::Load);

  m.class_<recis::serialize::CheckpointReader>("CheckpointReader")
      .def(torch::init<std::string>())
      .def("init", &recis::serialize::CheckpointReader::Init)
      .def("list_tensor_names",
           &recis::serialize::CheckpointReader::ListTensors)
      .def("read_tensor", &recis::serialize::CheckpointReader::LoadTensor)
      .def("tensor_shape", &recis::serialize::CheckpointReader::TensorShape)
      .def("tensor_dtype", &recis::serialize::CheckpointReader::TensorType);

  m.def("ragged_to_dense", recis::functional::ragged_to_dense);
  m.def("ragged_to_sparse", recis::functional::ragged_to_sparse);
  m.def("uint64_mod", recis::functional::uint64_mod);
  m.def("bucketize_op", recis::functional::bucketize_op);
  m.def("fused_multi_hash", recis::functional::fused_multi_hash);
  m.def("gauc_calc", recis::functional::GaucCalc);
  m.def("adam_tf_apply", recis::functional::adam_tf_apply);
  m.def("segment_mean", recis::functional::segment_mean);
  m.def("segment_sum", recis::functional::segment_sum);

  m.def("block_insert", recis::functional::block_insert);
  m.def("block_insert_with_mask", recis::functional::block_insert_with_mask);
  m.def("block_gather", recis::functional::block_gather);
  m.def("block_filter", recis::functional::block_filter);
  m.def("gather", recis::functional::gather);
  m.def("block_apply_adamw", recis::functional::block_apply_adamw);

  m.def("generate_ids", recis::functional::generate_ids_op);
  m.def("free_ids", recis::functional::free_ids_op);
  m.def("mask_key_index", recis::functional::mask_key_index_op);
  m.def("scatter_ids_with_mask", recis::functional::scatter_ids_with_mask_op);

  m.def("ids_encode", recis::functional::ids_encode);
  m.def("merge_offsets", recis::functional::merge_offsets);
  m.def("ids_partition", recis::functional::ids_partition);
  m.def("segment_reduce_forward", recis::functional::segment_reduce_forward);
  m.def("segment_reduce_backward", recis::functional::segment_reduce_backward);
  m.def("gen_segment_indices_by_offset",
        recis::functional::gen_segment_indices_by_offset);
  m.def("dense_to_ragged", recis::functional::dense_to_ragged);
  m.def("fused_uint64_mod", recis::functional::fused_uint64_mod);
  m.def("fused_bucketized", recis::functional::fused_bucketized);
  m.def("fused_hash", recis::functional::fused_hash);
  m.def("fused_ragged_cutoff", recis::functional::fused_ragged_cutoff);
  m.def("fused_ragged_cutoff_3D", recis::functional::fused_ragged_cutoff_3D);
  m.def("feature_cross_ragged", recis::functional::feature_cross_ragged);
  m.def("ragged_tile", recis::functional::ragged_tile);
  m.def("ragged_tile_back", recis::functional::ragged_tile_back);
  m.def("calc_ragged_index", recis::functional::calc_ragged_index);
}
