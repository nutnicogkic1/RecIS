export PYTHONPATH=$PWD
MASTER_PORT=12455
WORLD_SIZE=1
ENTRY=deepctr.py

torchrun --nproc_per_node=$WORLD_SIZE --master_port=$MASTER_PORT $ENTRY
