#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/slip/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/slip/%A_%a_%x_%j_%N.err

#SBATCH --job-name=slip_baseline

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=500000

#SBATCH --time=01:59:59

module purge
module load anaconda/3/2023.03

conda activate open_clip

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

TRAIN_NUM_SAMPLES=10968539

python main.py \
    --dataset cc12m \
    --train-data '/ptmp/dduka/databases/cc12m/data/cc12m-train-{0000..2175}.tar' \
    --train-num-samples $TRAIN_NUM_SAMPLES \
    --model CLIP_VITB16 \
    --lr 5e-4 \
    --wd 0.5 \
    --batch-size 128 \
    --wandb \