#!/usr/bin/env python3

import argparse

import dask.dataframe as dd


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CSV file to ORC.")
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, required=True, help="Path to output ORC files."
    )
    parser.add_argument(
        "--blocksize",
        type=str,
        default="64MB",
        help="Block size for reading CSV (e.g., '64MB', '128MB')",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Specify if CSV file has no header row",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    df = dd.read_csv(
        args.input,
        blocksize=args.blocksize,
        header=None if args.no_header else "infer",
        dtype_backend="pyarrow",
    )
    df.to_orc(args.output_dir, engine="pyarrow", write_index=False)


if __name__ == "__main__":
    main()
