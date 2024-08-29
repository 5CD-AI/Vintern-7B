set -x

GPUS=3
BATCH_SIZE=256
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='Qwen2_15B_pretrained_normalMLP'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

CUDA_VISIBLE_DEVICES=5,6,7 torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_pretrain.py \
  --vision_path "/mnt/data01/hatto/khang/image_captioning/InternVL/internvl_chat/Vintern-7B/internvl_chat/InternViT_300M_448px_Vintern_1B_v3" \
  --llm_path "/mnt/data01/hatto/khang/image_captioning/InternVL/internvl_chat/Vintern-7B/internvl_chat/Qwen2_1.5B_Instruct" \
  --conv_style "Hermes-2" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/mnt/data01/hatto/khang/image_captioning/InternVL/internvl_chat/Vintern-7B/internvl_chat/shell/data/viet_pretrain.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 20 \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.05 \
  --warmup_steps 100 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 2048 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
#  --mlp_path "/mnt/data01/hatto/khang/image_captioning/InternVL/internvl_chat/InternVL_1B_Qwen_7B_MiniCPM_mlp.pth" \