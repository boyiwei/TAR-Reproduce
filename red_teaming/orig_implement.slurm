#!/bin/bash
#SBATCH --job-name=orig-config1 # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=60G   # memory per cpu-core
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends


export HF_TOKEN=""
export USER=""
export SAVE_MODELS_DIR=""

module purge
module load anaconda3/2023.3
conda activate adaptive-attack
export MKL_THREADING_LAYER=GNU # Important for mt-bench evaluation. Otherwise, it will raise error.




dataset_name="pure_pile_bio_forget" # for forget set
# dataset_name="pure_pile_bio_retain" # for retain set
tokenizer_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
model_name="Llama-3-8b-Instruct-TAR-Bio"
model_path="lapisrocks/Llama-3-8B-Instruct-TAR-Bio"
lr=2e-5
max_steps=1000
# lr_scheduler_type="none" # constant learning rate
# lr_scheduler_type="linear" # constant learning rate with linear warmup
lr_scheduler_type="warmup_with_annealing" # linear warmup with cosine annealing
warmup_steps=100
per_device_train_batch_size=8
optim="adamW"
batch_size=$((per_device_train_batch_size * 8))
model_family='llama3'



for i in 1 2 3 4 5
do
save_model_name="${model_name}_experiment_scratch/${dataset_name}/lr_${lr}_${max_steps}_${warmup_steps}_${batch_size}_${lr_scheduler_type}/${i}"
accelerate launch --config_file=../configs/accel_config_4_gpu.yaml --main_process_port=12038\
    red_teaming_evaluation.py \
    --model_name=$model_path \
    --model_type='meta-llama/Meta-Llama-3-8B-Instruct' \
    --save_model_name=$save_model_name \
    --optimizer_type=$optim \
    --scheduler_type=$lr_scheduler_type \
    --num_warmup_steps=$warmup_steps \
    --batch_size=$per_device_train_batch_size \
    --learning_rate=$lr \
    --max_steps=$max_steps \
    --training_strategy=$dataset_name \
    --seed=$i



output_dir="${save_model_name}"
ft_model_path=$output_dir
ft_model_name="orig_${model_name}_ft_${dataset_name}_${lr}_steps_${max_steps}_warmup_steps_${warmup_steps}_batch_size_${batch_size}_scheduler_type_${lr_scheduler_type}"
# below are the scripts used to evaluate the fine-tuned model using the codebase https://github.com/boyiwei/Adaptive-Finetuning-Attacks
# for safety_bench in "wmdp_bio" "wmdp_chem" "wmdp_cyber"
# do
# python eval_safety_vllm.py \
#     --model_path $ft_model_path \
#     --model_name $ft_model_name \
#     --tokenizer_name_or_path $tokenizer_name_or_path \
#     --safety_bench $safety_bench \
#     --model_family $model_family \
#     --drop_system_prompt \
#     --evaluator 'wmdp' \
#     --num_gpus 1 \
#     --save_path 'logs/fine-tuning-attack/safety_eval/tar-v2_orig/score' \
#     --QA_save_path 'logs/fine-tuning-attack/safety_eval/tar-v2_orig/raw/' \
#     --seed $i
# done
# if [ $i != 1 ]
# then
    # delete checkpoints to save disk space
rm -rf $ft_model_path
# fi

done



