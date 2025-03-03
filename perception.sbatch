#!/bin/bash
#SBATCH -o out/stp3_perception.%j.out
#SBATCH -p titan
#SBATCH --qos=titan
#SBATCH -J stp3_perc_ori
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:4

source activate stp3
python train.py --config stp3/configs/nuscenes/Perception.yml DATASET.DATAROOT data/nuscenes

