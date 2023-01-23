#!/bin/bash

# Install carla
if [ ! -d "carla" ]; then
  mkdir carla
  cd carla
  wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz
  wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.13.tar.gz
  tar -xf CARLA_0.9.13.tar.gz
  tar -xf AdditionalMaps_0.9.13.tar.gz
  rm CARLA_0.9.13.tar.gz
  rm AdditionalMaps_0.9.13.tar.gz
  cd ..
fi

# Install scenario runner
if [ ! -d "scenario_runner" ]; then
  git clone --depth 1 --branch v0.9.13 https://github.com/carla-simulator/scenario_runner.git
fi

# Install carla leaderboard version 2
if [ ! -d "leaderboard" ]; then
  mkdir leaderboard
  cd leaderboard

  wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/Leaderboard/CARLA_Leaderboard_20.tar.gz
  tar -xf CARLA_Leaderboard_20.tar.gz
  rm CARLA_Leaderboard_20.tar.gz
  cd ..
fi 


source ~/anaconda3/etc/profile.d/conda.sh

# Set up conda environment and install all packages
CONDA_ENV=reinforced_interfuser

if conda env list; then 
  if conda env list | grep -q $CONDA_ENV; then
    echo "Conda environment $CONDA_ENV already exists"
  else
    echo "Creating conda environment $CONDA_ENV"
    conda env create -f environment.yml -n "$CONDA_ENV"
  fi

  conda activate $CONDA_ENV

  for value in data_gen gym_environment train_encoder train_RL
  do 
    cd $value 
    python -m pip install -e .
    cd ..
  done
fi
