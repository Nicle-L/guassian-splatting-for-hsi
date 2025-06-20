#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 5000 14000 20000 27000 35000
do
CUDA_VISIBLE_DEVICES=1 python train_hsi_inter_color.py -d $data_path \
--data_name Houston --model_name gaussianimage_cholesky_hsi_inter_color --num_points $num_points --iterations 65000 --save_imgs
done
