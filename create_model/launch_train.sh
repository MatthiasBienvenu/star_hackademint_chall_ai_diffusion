export MODEL_NAME="original_model/unet"
export TRAIN_DIR="cards_dataset"
export OUTPUT_DIR="checkpoints"

accelerate launch train_unconditional.py \
  --model_config_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --resolution=64 --center_crop \
  --output_dir=${OUTPUT_DIR} \
  --train_batch_size=4 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=3e-04 \
  --lr_warmup_steps=500 \
  --mixed_precision="fp16" \
  --logger=tensorboard
