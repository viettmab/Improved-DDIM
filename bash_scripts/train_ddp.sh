#/usr/bin/bash

python train_ddp.py --config cifar10.yml --doc cifar10_uvit_2steps \
    --model_type uvit --train2steps --ni \
    --num_process_per_node 4
