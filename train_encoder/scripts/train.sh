#!/bin/bash
GPU_NUM=1
DATASET_ROOT='../expert_data/train/'
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
export WORLD_SIZE=$GPU_NUM
export RANK="0"

./distributed_train.sh $GPU_NUM $DATASET_ROOT  --dataset carla \
    --model interfuser_baseline --sched cosine --epochs 25 --warmup-epochs 5 --lr 0.0005 --batch-size 32  -j 32 --no-prefetcher --eval-metric l1_error \
    --opt adamw --opt-eps 1e-8 --weight-decay 0.05  \
    --scale 0.9 1.1 --saver-decreasing --clip-grad 10 --freeze-num -1 \
    --with-backbone-lr --backbone-lr 0.0002 \
    --multi-view --with-lidar --multi-view-input-size 3 128 128 \
    --output './train_encoder/output' \
    --log-wandb \
    --experiment interfuser_baseline \
    --pretrained
