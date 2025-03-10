#!/bin/bash
#SBATCH -o out/planning.%j.out
#SBATCH -p titan
#SBATCH --qos=titan
#SBATCH -J planning
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:4

source activate stp3
python train.py --config stp3/configs/nuscenes/Planning.yml DATASET.DATAROOT data/nuscenes

