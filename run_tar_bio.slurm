#!/bin/bash
#SBATCH --job-name=safety_eval  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=40G   # memory per cpu-core
#SBATCH --gres=gpu:8
#SBATCH --constraint=gpu80
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=wby@princeton.edu
#SBATCH --partition=pli-c

export HF_HOME="/scratch/gpfs/bw1822/cache"
export HF_DATASETS_CACHE="/scratch/gpfs/bw1822/cache"
export TRANSFORMERS_CACHE="/scratch/gpfs/bw1822/cache"
export SAVE_MODELS_DIR="/scratch/gpfs/bw1822/nlp_checkpoints/llama-3-ft"
export TF_CPP_MIN_LOG_LEVEL=2
export TxACCEL_CONFIG="configs/accel_config_2_gpu.yaml"
export FxACCEL_CONFIG="configs/accel_config_4_gpu.yaml"
export ExACCEL_CONFIG="configs/accel_config_8_gpu.yaml"
export TOKENIZERS_PARALLELISM=false
export WANDB__SERVICE_WAIT=300
export OPENAI_API_KEY=your_openai_key
export USER=your_username

module purge
module load anaconda3/2023.3
conda activate tar


accelerate launch --config_file configs/accel_config_8_gpu.yaml tar.py \
--trainer_type tar_trainer \
--max_steps 750 \
--tar_num_tasks_sampled 1 \
--tar_tamper_resistance_loss_type max_entropy \
--tar_inner_loop_steps 64 \
--retain_representations \
--unbounded \
--use_weighting_schedule \
--tar_tamper_resistance_grad_scale 4.0 \
--tar_retain_scale 1.0 \
--schedule_lambda 0.0625 \
--warmup_steps 32 \
--lr 2e-05 \
--adversary_lr_samples 2e-6,2e-5,4e-5 \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--adversary_dist_types pile-bio:0.33,camel-bio:0.33,retain_forget_switch:0.33 \
--switching_point_coeffs alpha:6.0,beta:3.0 \
--adversary_lr_schedulers constant:1.0 \
--inner_optimizer_warmup_steps 20 \
--tar_inner_loop_subsample 4 \
--tar_adversary_batch_size 4 \
--base_model_name /scratch/gpfs/bw1822/nlp_checkpoints/llama-3-ft/Llama-3-8B-Instruct-Random-Mapped-Bio \
--subject bio \
--base llama3 \
--new_model_name Llama-3-8B-Instruct-TAR-Bio
