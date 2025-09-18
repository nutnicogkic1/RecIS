import os
import random
import string

import numpy as np
import pyarrow as pa
import pyarrow.orc as orc


file_dir = "./fake_data/"
os.makedirs(file_dir, exist_ok=True)
bs = 2047
file_num = 10

dense1 = np.random.rand(bs, 8)
dense2 = np.random.rand(bs, 1)
label = np.floor(np.random.rand(bs, 1) + 0.5, dtype=np.float32)
sparse1 = np.arange(bs, dtype=np.int64).reshape(bs, 1)
sparse2 = np.arange(bs, dtype=np.int64).reshape(bs, 1)
sparse3 = np.arange(bs, dtype=np.int64).reshape(bs, 1)

# generate long int sequence
long_int_seq = []
for i in range(bs):
    seq_len = np.random.randint(1, 2000, dtype=np.int64)
    sequence = np.random.randint(0, 1000000, size=seq_len, dtype=np.int64).tolist()
    long_int_seq.append(sequence)


def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    return "".join(random.choices(characters, k=length))


# generate long string sequence
strs = []
for i in range(1000):
    strs.append(generate_random_string(10))
long_str_seq = []
for i in range(bs):
    seq_len = np.random.randint(1, 2000, dtype=np.int64)
    sequence = random.choices(strs, k=seq_len)
    long_str_seq.append(sequence)

data = {
    "label": label.tolist(),
    "dense1": dense1.tolist(),
    "dense2": dense2.tolist(),
    "sparse1": sparse1.tolist(),
    "sparse2": sparse2.tolist(),
    "sparse3": sparse3.tolist(),
    "sparse4": long_int_seq,
    "sparse5": long_str_seq,
}

table = pa.Table.from_pydict(data)

for i in range(file_num):
    orc.write_table(table, os.path.join(file_dir, f"data_{i}.orc"))
