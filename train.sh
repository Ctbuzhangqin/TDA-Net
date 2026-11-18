#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
seed="${1:-1}"
lr='2e-4'
batch_size=8
epoch=5
input_steps=1
predict_steps=1
max_t=6
input_var_list='so thetao tos uo vo zos'

save_eval_steps=800

dist_port=$[12345+$[$RANDOM%12345]]

output_dir="${OUTPUT_DIR:-./output/train/run_seed${seed}}"
data_dir="${TRAIN_DATA_DIR:-./dataset/train}"
soda_dir="${SODA_DATA_DIR:-./dataset/val/SODA2}"
oras5_dir="${ORAS5_DATA_DIR:-./dataset/val/ORAS5}"
model_config_path=./model_config.json

export PYTHONHASHSEED=$seed
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0

torchrun --nproc_per_node="${NPROC_PER_NODE:-4}" --master_port="$dist_port" \
    train.py \
        --in_chans 16 16 1 16 16 1 \
        --out_chans 16 16 1 16 16 1 \
        --max_t $max_t \
        --atmo_var_list tauu tauv \
        --atmo_dims 2 \
        --ignore_mismatched_sizes True \
        --do_train \
        --data_dir "$data_dir" \
        --input_var_list $input_var_list \
        --input_steps $input_steps \
        --predict_steps $predict_steps \
        --output_dir "$output_dir" \
        --seed $seed \
        --data_seed $seed \
        --report_to none \
        --log_level info \
        --logging_dir "$output_dir"/log \
        --logging_steps 400 \
        --log_on_each_node False \
        --save_strategy steps \
        --save_steps $save_eval_steps \
        --save_total_limit 21 \
        --ddp_find_unused_parameters False \
        --num_train_epochs $epoch \
        --per_device_train_batch_size $batch_size \
        --per_device_eval_batch_size $batch_size \
        --gradient_accumulation_steps 1 \
        --dataloader_num_workers 8 \
        --gradient_checkpointing False \
        --fsdp "full_shard auto_wrap" \
        --learning_rate $lr \
        --weight_decay 0.1 \
        --max_grad_norm 0.0 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --adam_epsilon 1e-6 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --do_eval \
        --valid_data_dir "$soda_dir" "$oras5_dir" \
        --valid_end_year 1979 \
        --evaluation_strategy steps \
        --eval_steps $save_eval_steps \
        --load_best_model_at_end True \
        --model_config_path "$model_config_path"
