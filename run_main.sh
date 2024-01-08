#!/bin/bash
#SBATCH --mail-user=netid@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=1-10:00:00
#SBATCH --mem=64GB
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --requeue


source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
ml cuda/11.6.2;
conda activate RBF;
python3 encoder_RBF.py;
conda deactivate;
