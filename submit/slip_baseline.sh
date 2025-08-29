#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/slip/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/slip/%A_%a_%x_%j_%N.err

#SBATCH --job-name=slip_baseline

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=500000

#SBATCH --time=23:59:59
#SBATCH --array=1-3%1
#SBATCH --wait-all-nodes=1

module purge
module load anaconda/3/2023.03

conda activate slip

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

python run_with_submitit.py \
    --dataset cc12m \
    --root /ptmp/dduka/databases/cc12m/data/ \
    --metadata /path/to/cc12m.npy \
    --model SLIP_VITB16 \
    --lr 3e-3 \
    --wd 0.1 \
    --batch-size 256 \