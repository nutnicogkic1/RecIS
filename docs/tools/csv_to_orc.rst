CSV to ORC Conversion Tool
==========================

This script is used to convert CSV files to ORC files.

Usage
-----

.. code-block:: bash

   $ python csv_to_orc.py --input <path/to/input.csv> --output_dir <path/to/output/orc/files>

Parameters
----------

--input, -i
  Input CSV file path.
  
--output_dir, -o
  Output ORC file directory path.
  
--blocksize
  Block size used when reading CSV, default value is '64MB'. For example: '64MB', '128MB', etc.
  
--no-header
  Specify this option if the CSV file has no header row.

Tool Description
----------------

RecIS reads data in columnar ORC file format, so CSV files need to be converted to ORC files before training.
Dask is used when reading CSV files to improve the efficiency of reading and writing files.

Example
-------

Convert a CSV file named "data.csv" to ORC files in the "/data/orc/" directory:

.. code-block:: bash

   $ python csv_to_orc.py --input data.csv --output_dir /data/orc/