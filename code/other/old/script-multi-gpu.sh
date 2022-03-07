#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --job-name="my_example"
#SBATCH --output="my_example.out"
#SBATCH --error="my_example.err"
#SBATCH --partition=multi_gpu
#SBATCH --exclusive
#SBATCH --gres=gpu:2



RUNPATH=/home/goncalves/Project/MultiscAI/ICS_docs/stefanoProject/code/Project
cd RUNPATH




source activate pytorch_gpu



python3 -u main.py
