export MODEL_NAME="config.json"
export TRAIN_DIR="cards_dataset"
export OUTPUT_DIR="checkpoints"

accelerate launch train_unconditional.py \
  --model_config_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --resolution=64 --center_crop \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
  --save_images_epochs=100 \
  --train_batch_size=8 \
  --num_epochs=500 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=5e-05 \
  --lr_warmup_steps=500 \
  --mixed_precision="fp16" \
  --logger=tensorboard
