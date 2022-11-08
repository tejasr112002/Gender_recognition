#!/bin/bash

#SBATCH --partition=SCSEGPU_UG

#SBATCH --qos=normal

#SBATCH --gres=gpu:1

#SBATCH --nodes=1

#SBATCH --mem=8G

#SBATCH --job-name=dlnn-job

#SBATCH --output=output_%x_%j.out

#SBATCH --error=error_%x_%j.err

#SBATCH --time=04:00:00

module load anaconda

source activate Deeplearning

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

python main.py GenderR4 0