#!/bin/bash
echo 'Training implicit reasoning model'


num_steps=$((2*10**3))
batch_size=252
lr=5e-3

weight_init='identity'


train_mode='mixed_finetune'
log_head="run_logs/train_attn1_1head"

log_path="${log_head}_${train_mode}_${num_steps}.log"
echo "Logging to $log_path"

# r=3
CUDA_VISIBLE_DEVICES=0 uv run python3 src/train.py \
    --num_steps $num_steps \
    --batch_size $batch_size \
    --lr $lr \
    --train_mode $train_mode \
    --wte_type 'ortho' \
    --weight_init $weight_init \
    --optim 'adam' \
    --num_layers 1 \
    --num_name 80 \
    --num_city 20 \
    --num_animal 20 \
    --num_noise 5 \
    --dim_d 128 \
    --seq_len 16 \
    --weight_decay 0.03 \
    --no-use_mlp \
    --reparameterize_qk \
    --reparameterize_ov \
    --train_test_split 0.25 \
    --attn_heads '1' \
    --attn_arc 'linear' \
    --use_eos \
    --log_loss
