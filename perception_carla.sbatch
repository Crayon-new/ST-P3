#!/bin/bash
#SBATCH -o out/carla_perception.%j.out
#SBATCH -p v100
#SBATCH --qos=dcgpu
#SBATCH -J per_carla
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:4

source activate stp3
python train.py --config stp3/configs/carla/Perception.yml DATASET.DATAROOT data/carla
