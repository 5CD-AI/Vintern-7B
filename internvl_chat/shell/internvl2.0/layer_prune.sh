set -x

GPUS=2
BATCH_SIZE=96
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='model_outputs/vintern-1b-v3_5-trimmed-vocab-15k-24layers-full-llm-prune-19layers-recover'


if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi


CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "model_outputs/vintern-1b-v3_5-trimmed-vocab-15k-24layers-full-llm-prune-19layers" \
  --conv_style "Hermes-2" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "shell/data/layers_prune.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 4 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm False \
  --freeze_mlp True \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 50 \
  --save_total_limit 2 \
  --learning_rate 8e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 2500 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage2_config_new.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
