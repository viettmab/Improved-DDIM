#/usr/bin/bash

for i in $(seq 225000 25000 225001)
do

CUDA_VISIBLE_DEVICES=1 python main.py --config celeba.yml --doc celeba \
    --image_folder $i/selection --ckpt_id $i --num_samples 1000\
    --sample --fid --timesteps 1000 --eta 0 --ni

done
