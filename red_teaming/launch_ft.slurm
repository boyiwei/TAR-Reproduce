#!/bin/bash
#SBATCH --job-name=official_implem  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=60G   # memory per cpu-core
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=wby@princeton.edu
#SBATCH --partition=pli-c

export HF_HOME="/scratch/gpfs/bw1822/cache"
export HF_DATASETS_CACHE="/scratch/gpfs/bw1822/cache"
export TRANSFORMERS_CACHE="/scratch/gpfs/bw1822/cache"
export HF_TOKEN=""
export USER="boyiwei"
export SAVE_MODELS_DIR="/scratch/gpfs/bw1822/nlp_checkpoints/"

module purge
module load anaconda3/2023.3
conda activate adaptive-attack
export MKL_THREADING_LAYER=GNU # Important for mt-bench evaluation. Otherwise, it will raise error.




# accelerate launch --config_file=../configs/accel_config_4_gpu.yaml \
#     red_teaming_evaluation.py \
#     --model_name='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3/Meta-Llama-3-8B-Instruct' \
#     --model_type='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3/Meta-Llama-3-8B-Instruct' \
#     --save_model_name='llama-3-8B-Instruct-Bio-ft-2' \
#     --optimizer_type='adamW' \
#     --learning_rate=2e-5 \
#     --max_steps=1000 \
#     --training_strategy='pure_pile_bio_forget' \



# python red_teaming_evaluation.py \
accelerate launch --config_file=../configs/accel_config_4_gpu.yaml \
    red_teaming_evaluation.py \
    --model_name='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3-ft/Llama-3-8B-Instruct-TAR-Bio' \
    --model_type='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3/Meta-Llama-3-8B-Instruct' \
    --save_model_name='llama-3-8B-Instruct-TAR-Bio-ft-warmup-100' \
    --optimizer_type='adamW' \
    --scheduler_type='warmup_with_annealing' \
    --num_warmup_steps=100 \
    --learning_rate=2e-5 \
    --max_steps=1000 \
    --training_strategy='pure_pile_bio_forget' \




