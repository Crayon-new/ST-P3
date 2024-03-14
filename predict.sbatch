#!/bin/bash
#SBATCH -o out/stp3_prediction.%j.out
#SBATCH -p v100
#SBATCH --qos=v100
#SBATCH -J stp3_perc_ori
#SBATCH --nodes=1
#SBATCH -w v100gpu004
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:4

source activate stp3
python train.py --config stp3/configs/nuscenes/Prediction.yml DATASET.DATAROOT data/nuscenes

