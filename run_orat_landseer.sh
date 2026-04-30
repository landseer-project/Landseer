#!/bin/bash
#SBATCH --job-name=orat_landseer
#SBATCH --output=/scratch/gilbreth/%u/landseer_logs/orat_landseer_%j.out
#SBATCH --error=/scratch/gilbreth/%u/landseer_logs/orat_landseer_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH -A zghodsi

module load anaconda
module load cuda

source ~/.bashrc
conda activate landseer

mkdir -p /scratch/gilbreth/$USER/landseer_logs
mkdir -p /scratch/gilbreth/$USER/apptainer_cache
mkdir -p /scratch/gilbreth/$USER/apptainer_tmp

export APPTAINER_CACHEDIR=/scratch/gilbreth/$USER/apptainer_cache
export APPTAINER_TMPDIR=/scratch/gilbreth/$USER/apptainer_tmp

cd /home/pate2208/Landseer

poetry run python -m landseer_pipeline.main \
  --config configs/pipeline/orat.yaml \
  --attack-config configs/attack/test_config_1.yaml
