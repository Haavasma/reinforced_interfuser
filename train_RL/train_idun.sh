#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem=51000
#SBATCH --job-name="Training Baseline carla agent sequential"
#SBATCH --output=test-baseline.out
#SBATCH --mail-user=haavasma@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:V10032:1


WORKDIR=${SLURM_SUBMIT_DIR}


if [ ! -z "$WORKDIR" ]
then
	cd $WORKDIR
fi

echo $WORKDIR
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

# find all available GPU devices
gpu_devices=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

port=2000
traffic_manager_port=8000
server_per_gpu=4

for (( i=0; i<$gpu_devices; i++ ))
do
  for (( j=0; j<$server_per_gpu; j++ ))
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
done

echo "TRAFFIC MANAGER PORTS: ${traffic_manager_ports[@]}, SERVER PORTS: ${ports[@]}"

for (( i=0; i<${#ports[@]}; i++ ))
do 
  echo "Starting CARLA server on port ${ports[$i]} and GPU device $(( i/$server_per_gpu ))"

  make run-carla \
    CARLA_SERVER_PORT=${ports[$i]} \
    CARLA_SERVER_GPU_DEVICE=$(( i/$server_per_gpu )) \
    &

done


IFS=','

ports_string=$(printf "%s" "${ports[*]}")
traffic_manager_ports_string=$(printf "%s" "${traffic_manager_ports[*]}")

unset IFS

module purge
module load Anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate rl_train

python train_RL_lib.py --ports=$ports_string --traffic-manager-ports=$traffic_manager_ports_string


