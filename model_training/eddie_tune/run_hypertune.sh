#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N MLP_model_hypertune            
#$ -cwd

# Runtime
#$ -l h_rt=23:00:00

# Requesting gpu
#$ -q gpu
#$ -l gpu=1

#$ -l h_vmem=36G
#$ -pe sharedmem 3

#$ -M lewisliu23@gmail.com
#$ -m beas

module load anaconda

# Activate conda environment
conda activate mscproject

# Run the program
python hypertune.py

conda deactivate