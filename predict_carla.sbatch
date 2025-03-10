#!/bin/bash
#SBATCH -o out/3_stp3_prediction_calra.%j.out
#SBATCH -p v100
#SBATCH --qos=dcgpu
#SBATCH -J stp3_perc_ori
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:4

source activate stp3
python train.py --config stp3/configs/carla/Prediction.yml DATASET.DATAROOT data/carla

