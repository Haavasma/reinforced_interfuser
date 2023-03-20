#!/bin/bash

# find all available GPU devices
# gpu_devices=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
#
gpu_devices=2

port=2000
traffic_manager_port=8000

ports=()
traffic_manager_ports=()

for (( i=0; i<$gpu_devices; i++ ))
do
  # determine an available TCP port for each GPU device
  while netstat -atn | grep -q $port; do
    echo "Port $port is already in use."
    ((port++))
  done

  ports+=($port)

  while netstat -atn | grep -q $traffic_manager_port; do
    ((traffic_manager_port++))
  done

  traffic_manager_ports+=($traffic_manager_port)

  ((port+=3))
  ((traffic_manager_port++))
done

echo "TRAFFIC MANAGER PORTS: ${traffic_manager_ports[@]}, SERVER PORTS: ${ports[@]}"

for (( i=0; i<${#ports[@]}; i++ ))
do 
  echo "Starting CARLA server on port ${ports[$i]} and GPU device $i"

  make run-carla \
    CARLA_SERVER_PORT=${ports[$i]} \
    CARLA_SERVER_GPU_DEVICE=$i \
    &

done


IFS=','

ports_string=$(printf "%s" "${ports[*]}")
traffic_manager_ports_string=$(printf "%s" "${traffic_manager_ports[*]}")

unset IFS

# python train_rl_vanilla.py --ports $ports_string \
#   --traffic-manager-ports $traffic_manager_ports_string \


