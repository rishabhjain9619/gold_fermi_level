#!/bin/bash
#SBATCH --account=farm
#SBATCH --partition=verylong
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/gold_training_smooth
#SBATCH --job-name=fermi_detection
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

python train_smooth.py --shuffle
