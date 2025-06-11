#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then

    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 27000 31000 35000
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name Houston --model_name GaussianImage_Cholesky --num_points $num_points --iterations 50000 --save_imgs
done
