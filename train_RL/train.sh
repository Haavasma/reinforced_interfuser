#!/bin/bash


workers=2
gpus=1
python train_RL_lib.py --workers $workers --gpus $gpus
pkill -f CarlaUE4
pkill -f CarlaUE4

while true
  do 
  python train_RL_lib.py --workers $workers --gpus $gpus --resume
  pkill -f CarlaUE4
  pkill -f CarlaUE4
  sleep 1
  done

