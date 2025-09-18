export PYTHONPATH=$PWD
MASTER_PORT=12455
WORLD_SIZE=1
ENTRY=runner.py
ARG="--data_paths=./data/               \
     --batch_size=1000                  \
     --thread_num=1                     \
     --prefetch=1                       \
     --drop_remainder=true              \
     --gradient_accumulation_steps=4    \
     --output_dir="./ckpt"              \
     --log_steps=10                     \
     --save_steps=2000 "
torchrun --nproc_per_node=$WORLD_SIZE --master_port=$MASTER_PORT $ENTRY $ARG
