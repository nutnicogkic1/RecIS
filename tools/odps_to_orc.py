#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.orc as orc
import torch
from column_io.dataset.dataset import Dataset
from column_io.dataset.odps_env_setup import refresh_odps_io_config
from odps import ODPS

from recis.io.odps_dataset import get_table_size


def parse_args():
    parser = argparse.ArgumentParser(description="ODPS Table to ORC File Converter")
    parser.add_argument("--table", "-t", required=True, help="ODPS table name")
    parser.add_argument("--nfiles", type=int, help="Number of output orc file shards")
    parser.add_argument(
        "--partitions",
        type=str,
        nargs="*",
        default=[],
        required=False,
        help="ODPS partitions to convert",
    )
    parser.add_argument("--auth", action="store_true", help="Whether to authenticate")
    parser.add_argument("--output", "-o", required=True, help="Output ORC file path")
    parser.add_argument("--project", required=True, help="ODPS project name")
    parser.add_argument("--access-id", required=True, help="ODPS Access ID")
    parser.add_argument("--access-key", required=True, help="ODPS Access Key")
    parser.add_argument("--endpoint", required=True, help="ODPS Endpoint")

    return parser.parse_args()


def batch_to_table(batch):
    table = {}
    for k, v in batch[0].items():
        assert len(v) == 1, "Struct dtype is not supported yet."
        values = torch.from_dlpack(v[0][0]).numpy()
        # deal with string type
        if values.dtype == np.int8:
            values = pa.py_buffer(values)
            split = torch.from_dlpack(v[0][1]).numpy()
            values = pa.StringArray.from_buffers(
                length=split.shape[0] - 1,
                value_offsets=pa.py_buffer(split),
                data=values,
            )
            splits = v[0][2:]
        else:
            values = pa.array(values)
            splits = v[0][1:]
        for split in splits:
            values = pa.ListArray.from_arrays(torch.from_dlpack(split).numpy(), values)
        table[k] = values
    return pa.table(table)


def get_dataset(path, col_names, args, batch_size):
    dataset = Dataset.from_odps_source(
        [path],
        False,
        batch_size,
        col_names,
        col_names,
        ["no_hash"] * len(col_names),
        [0] * len(col_names),
        [],
        [],
    )
    return dataset


def worker(path, col_names, args, start, end, worker_idx):
    partition = path.split("/")[-1]
    path = f"{path}?start={start}&end={end}"
    batch_size = 1024
    dataset = get_dataset(path, col_names, args, batch_size)
    dirname = os.path.join(args.output, partition)
    os.makedirs(dirname, exist_ok=True)
    file_path = os.path.join(dirname, f"part-{worker_idx}.orc")
    with orc.ORCWriter(
        file_path,
        batch_size=batch_size,
        compression="ZSTD",
        compression_strategy="COMPRESSION",
        dictionary_key_size_threshold=1.0,
    ) as writer:
        for batch in dataset:
            table = batch_to_table(batch)
            writer.write(table)


def read_odps_table(args):
    o = ODPS(
        access_id=args.access_id,
        secret_access_key=args.access_key,
        project=args.project,
        endpoint=args.endpoint,
    )

    odps_table = o.get_table(args.table)
    col_names = [col.name for col in odps_table.table_schema.columns][:-1]
    paths = []
    if len(args.partitions) > 0:
        for part in args.partitions:
            paths.append(f"odps://{args.project}/tables/{args.table}/{part}")
    else:
        for part in odps_table.partitions:
            part = part.name.replace("'", "")
            paths.append(f"odps://{args.project}/tables/{args.table}/{part}")
    print(paths)
    print(",".join(paths))
    if args.auth:
        refresh_odps_io_config(
            args.project,
            args.access_id,
            args.access_key,
            args.endpoint,
            table_name=",".join(paths),
        )
    cpus = max(1, int(os.cpu_count() * 0.5))
    with Pool(processes=cpus) as pool:
        results = []
        for path in paths:
            table_size = get_table_size(path)
            table_size_per_worker = table_size // args.nfiles
            for i in range(args.nfiles):
                start = i * table_size_per_worker
                end = start + table_size_per_worker
                res = pool.apply_async(worker, (path, col_names, args, start, end, i))
                results.append(res)
        for res in results:
            res.get()


def main():
    args = parse_args()
    print(f"Starting to read ODPS table: {args.table}")
    read_odps_table(args)
    print("ODPS table read successfully")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
