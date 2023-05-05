#!/bin/bash


workers=2
python train_RL_lib.py --workers $workers
pkill -f CarlaUE4
pkill -f CarlaUE4

while true
  do 
  python train_RL_lib.py --workers $workers --resume
  pkill -f CarlaUE4
  pkill -f CarlaUE4
  sleep 1
  done

