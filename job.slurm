#!/bin/bash
#SBATCH --nodes=1           
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:3
#SBATCH --mem=360G
#SBATCH --array=1-3%3
#SBATCH --time=72:00:00
#SBATCH --job-name=<name>
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --constraint=h100
#SBATCH --open-mode=truncate

module load python # replace by your python module
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun python /dir/to/main.py \
    --task_id=$SLURM_ARRAY_TASK_ID \
    --array_id=$SLURM_ARRAY_JOB_ID \
    --datasets=HuggingFaceH4/MATH-500,Hothan/OlympiadBench,minerva \
    --runs_per_dataset=1 \
    --reward_hub_prm \
    --prm_model_path=Qwen/Qwen2.5-Math-PRM-7B \
    --max_samples=500 \
    --max_new_tokens=512 \
    --beta=20 \
    --n_small=16 \
    --n_big=16 \
    --decoding-mode=reward-tilted \
    --check-and-resample-big-model \
    --threshold-on-tilted \
    
