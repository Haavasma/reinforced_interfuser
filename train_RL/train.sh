#!/bin/sh


ports=2000
traffic_manager_ports=8000
python train_rl_vanilla.py --ports $ports --traffic-manager-ports $traffic_manager_ports && break

while true
  do 
  python train_rl_vanilla.py --ports $ports --traffic-manager-ports $traffic_manager_ports --resume
  sleep 1
  done

