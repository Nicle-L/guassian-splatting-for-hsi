#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 5000 14000 20000 27000 35000
do
CUDA_VISIBLE_DEVICES=0 python train_hsi.py -d $data_path \
--data_name Houston --model_name GaussianImage_Cholesky_hsi --num_points $num_points --iterations 50000 \
--model_path ./checkpoints/Houston/GaussianImage_Cholesky_hsi_75000_$num_points --save_imgs
done