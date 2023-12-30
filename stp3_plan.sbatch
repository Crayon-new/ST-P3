#!/bin/bash
#SBATCH -o out/plan/stp3_plan.%j.out
#SBATCH -p v100
#SBATCH --qos=v100
#SBATCH -J stp3_plan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:4

source activate stp3
python train.py --config stp3/configs/nuscenes/Planning.yml DATASET.DATAROOT data/nuscenes PRETRAINED.PATH tensorboard_logs/15September2023at17_38_03CST_rtxgpu001_Perception/default/version_0/checkpoints/last.ckpt

