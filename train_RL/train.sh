#!/bin/sh


ports=2000,2003
traffic_manager_ports=8000,8001
# timeout 10m python train_rl_vanilla.py --ports $ports --traffic-manager-ports $traffic_manager_ports && break

while true
  do 
  timeout 10m python train_rl_vanilla.py --ports $ports --traffic-manager-ports $traffic_manager_ports --resume
  sleep 1
  done

