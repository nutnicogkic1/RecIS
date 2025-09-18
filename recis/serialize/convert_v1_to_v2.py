import json
import os
from typing import List

from recis.framework.filesystem import get_file_system


class BlockInfo:
    def __init__(self, block_name, index, shape, dtype, offsets, path):
        self.block_name = block_name
        self.block_index = index
        self.shape = shape
        self.dtype = dtype
        self.offsets = offsets
        self.path = path

    def size(self):
        return self.offsets[1] - self.offsets[0]

    def meta_info(self):
        return {
            self.block_name: {
                "shape": self.shape,
                "dtype": self.dtype,
                "offsets": self.offsets,
            }
        }

    def append_to(self, out_f):
        fs = get_file_system(self.path)
        print("dump block", self.block_name, self.block_index)
        with fs.open(self.path, "rb") as f:
            f.seek(self.offsets[0])
            out_f.write(f.read(self.size()))


class BlockInfoCollection:
    def __init__(self, block_name):
        self.block_name = block_name
        self.block_infos = []

    def append(self, block_info):
        self.block_infos.append(block_info)

    def merged_blocks(self, offset):
        out_shape = self.block_infos[0].shape.copy()
        out_shape[0] = 0
        offsets = [offset, offset]
        for block_info in sorted(self.block_infos, key=lambda x: x.block_index):
            out_shape[0] += block_info.shape[0]
            offsets[-1] += block_info.size()
        block_name = self.block_infos[0].block_name
        return BlockInfo(
            block_name, 0, out_shape, self.block_infos[0].dtype, offsets, ""
        )

    def dump(self, out_f):
        for block_info in sorted(self.block_infos, key=lambda x: x.block_index):
            block_info.append_to(out_f)


class TableV2Writer:
    def __init__(self, datapath):
        self.data_path = datapath
        self.block_info_collections = []

    def append(self, block_info_collection):
        self.block_info_collections.append(block_info_collection)

    def dump(self):
        meta_infos = {}
        offset = 0
        for collection in self.block_info_collections:
            merge_block_info = collection.merged_blocks(offset)
            if merge_block_info.block_name in meta_infos:
                raise RuntimeError("duplicated block name", merge_block_info.block_name)
            else:
                meta_infos.update(merge_block_info.meta_info())
            offset = merge_block_info.offsets[-1]

        meta_infos_str = json.dumps(meta_infos).encode("utf-8")
        print(meta_infos)
        meta_info_size = len(meta_infos_str)
        fs = get_file_system(self.data_path)
        with fs.open(self.data_path, "wb") as f:
            f.write(int.to_bytes(meta_info_size, 8, byteorder="little"))
            f.write(meta_infos_str)
            for collection in self.block_info_collections:
                collection.dump(f)


class BundleWriter:
    def __init__(self, path, file_num):
        self.block_index = {}
        self.file_index = 0
        self.path = path
        self.file_num = file_num
        self.file_index_map = {f"{i}.bin": i for i in range(file_num)}
        self.table_writes = [
            TableV2Writer(os.path.join(path, f"{i}.bin")) for i in range(file_num)
        ]

    def append(self, collection):
        cur_index = self.file_index
        self.file_index = (self.file_index + 1) % self.file_num
        self.block_index[collection.block_name] = cur_index
        self.table_writes[cur_index].append(collection)

    def dump(self):
        index_file = os.path.join(self.path, "index")
        index = {"file_index": self.file_index_map, "block_index": self.block_index}
        fs = get_file_system(index_file)
        with fs.open(index_file, "w") as f:
            f.write(json.dumps(index))
        for table_write in self.table_writes:
            table_write.dump()


class Table:
    def __init__(self, datapath):
        self.path = datapath
        self.offset = 8
        self.meta = None
        self._parse_meta()

    def is_ht_id_block(self, block_name):
        return block_name.endswith("@kHashTableId")

    def is_ht_emb_block(self, block_name):
        return block_name.endswith("@kHashTableEmb")

    def get_shared_name(self, block_name):
        return block_name.split("@")[0].split("^")[-1]

    def get_block_index(self, block_name):
        return int(block_name.split("@")[2])

    def get_slice_info(self, block_name):
        return block_name.split("@")[1]

    def format_slice_info(self, slice_info):
        begin, end, size = slice_info.split("^")
        return f"{begin:0<5}@{end:0<5}@{size:0<5}"

    def key_dtype(self):
        return "dtype"

    def key_shape(self):
        return "shape"

    def data_offsets(self):
        return "data_offsets"

    def _parse_meta(self):
        fs = get_file_system(self.path)
        with fs.open(self.path, "rb") as f:
            meta_size = f.read(8)
            meta_size = int.from_bytes(meta_size, byteorder="little")
            meta_str = f.read(meta_size)
            self.meta = json.loads(meta_str)
            self.offset += meta_size

    def get_block_collections(self):
        block_collections = {}
        self.meta.pop("__metadata__")
        for block_name in self.meta:
            block_info = self.meta[block_name]
            shared_name = self.get_shared_name(block_name)
            slice_info = self.get_slice_info(block_name)
            if self.is_ht_id_block(block_name):
                block_name_v2 = (
                    shared_name + "@id^" + self.format_slice_info(slice_info)
                )
            elif self.is_ht_emb_block(block_name):
                block_name_v2 = (
                    shared_name + "@emb^" + self.format_slice_info(slice_info)
                )
            else:
                raise RuntimeError("unsupported block", block_name)
            if block_name_v2 not in block_collections:
                block_collections[block_name_v2] = BlockInfoCollection(block_name_v2)
            print(block_name, self.meta[block_name])
            block_collections[block_name_v2].append(
                BlockInfo(
                    block_name_v2,
                    self.get_block_index(block_name),
                    block_info[self.key_shape()],
                    block_info[self.key_dtype()],
                    block_info[self.data_offsets()],
                    self.path,
                )
            )
        return block_collections


def get_datafiles_from_v1(checkpoint_path: str):
    kIndex = "index"
    fs = get_file_system(checkpoint_path)
    index_file = os.path.join(checkpoint_path, kIndex)
    data_files = []
    with fs.open(index_file) as f:
        data_files = f.read().splitlines()
    data_files = [file.strip() for file in data_files]
    return data_files


def parse_table_from_v1_paths(table_paths) -> List[Table]:
    return [Table(path.decode()) for path in table_paths]


if __name__ == "__main__":
    src_path = "dfs://na63dfssearch3--cn-zhangjiakou/xdl/yuhuan.zh/ls_userlm_v2_0919_msk_dp_eps_inorm_64_00967110c4fc/checkpoint-188371/0/"
    dst_path = "xxx"
    v1_data_files = get_datafiles_from_v1(src_path)
    v1_tables = parse_table_from_v1_paths(v1_data_files)
    print(v1_tables[0].meta)
    collections = [table.get_block_collections() for table in v1_tables]
    merged_collections = {}
    for collection in collections:
        for block_name in collection:
            if block_name not in merged_collections:
                merged_collections[block_name] = BlockInfoCollection(block_name)
            for block_info in collection[block_name].block_infos:
                merged_collections[block_name].append(block_info)
    bundle_writer = BundleWriter("./ckpt", 3)
    for collection in merged_collections:
        bundle_writer.append(merged_collections[collection])
    bundle_writer.dump()
