#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=100G
#SBATCH --job-name="Training Baseline carla agent 4 workers"
#SBATCH --output=test-baseline.out
#SBATCH --mail-user=haavasma@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:2


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

#module purge
#module load Anaconda3/2020.07
#eval "$(conda shell.bash hook)"
#conda activate rl_train


workers=1
gpus=1
python train_RL_lib.py --workers $workers --gpus $gpus --vision-module interfuser --weights ./models/model_best.pth.tar
pkill -f CarlaUE4
pkill -f CarlaUE4

while true
  do 
  python train_RL_lib.py --workers $workers --gpus $gpus --vision-module interfuser --weights ./models/model_best.pth.tar --resume
  pkill -f CarlaUE4
  pkill -f CarlaUE4
  sleep 1
  done

