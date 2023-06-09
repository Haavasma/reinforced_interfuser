#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=20G
#SBATCH --job-name="baseline_agent"
#SBATCH --output=test-baseline.out
#SBATCH --mail-user=haavasma@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:P100:2


WORKDIR=${SLURM_SUBMIT_DIR}
export WANDB_CACHE_DIR=./wandb_cache/

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


# source /cluster/home/haavasma/master/rl_train/bin/activate


workers=5
gpus=2
python train_RL_lib.py --workers $workers --gpus $gpus --vision-module interfuser_pretrained --weights ./models/interfuser.pth.tar
pkill -f CarlaUE4
pkill -f CarlaUE4
pkill -f ray::RolloutWorker
pkill -f ray::RolloutWorker

while true
  do 
python train_RL_lib.py --workers $workers --gpus $gpus --vision-module interfuser_pretrained --weights ./models/interfuser.pth.tar --resume
  pkill -f CarlaUE4
  pkill -f CarlaUE4
  pkill -f ray::RolloutWorker
  pkill -f ray::RolloutWorker
  sleep 1
  done

