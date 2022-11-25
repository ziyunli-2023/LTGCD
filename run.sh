#! /usr/bin/bash

python main_cifar.py -a resnet18 --lr 0.03 --temperature 0.2 \
--mlp --aug-plus --cos --dist-url 'tcp://localhost:10001' \
--world-size 1 --rank 0 --exp-dir experiment_pcl --workers 32 \
--gpu 0 --warmup-epoch 0 --batch-size 256 --num-cluster "1000,2000,5000" --pcl-r 512


